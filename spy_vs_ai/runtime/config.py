"""The canonical runtime configuration (FINAL-2 spec 12, 13, 20, 34).

    GameRuntimeConfig: level, seed, mode, view, speed, telemetry, time limit, black / white AgentRuntimeConfig
    AgentRuntimeConfig: policy, personality, provider?, model?, identity?, provider_options?, max_provider_calls?

`mode` is the execution semantics (spec 20):
    evaluation - the frozen research execution semantics of Stages 3-5: evaluation decision timing (the world stops
                 while a model thinks), the simulated mission clock (model latency spends no mission time) and the
                 seeded commit order. A runtime match under it is still a single match, not a scientific result.
    playable   - smoother live play: the mission clock runs on real time; real-LLM spies decide asynchronously
                 (live_async: the world keeps moving while a model thinks). A hybrid spy is validated only with
                 evaluation decision timing (provider latency spends no mission time), so with a hybrid spy the world
                 and the mission clock pause while a model thinks. Execution semantics differ from the frozen
                 evaluation protocol.
`view` is the display (spec 15-17): play (the game first) or research (decision inspector, history, telemetry).

The configuration is validated once, before anything starts (spec 13); invalid combinations are rejected.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from spy_vs_ai.runtime import policy_registry as pr
from spy_vs_ai.runtime.provider_registry import PROVIDER_IDS, PROVIDERS

LEVELS = ("level_01", "level_02", "level_03")
MODES = ("evaluation", "playable")
VIEWS = ("play", "research")
SPEEDS = (0.5, 1.0, 2.0)
TIME_LIMIT_RANGE = (30.0, 1800.0)
ACTORS = ("black", "white")
DEFAULT_OUTPUT_DIR = "runs"
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_SECRET_KEY_RE = re.compile(r"(api[_-]?key|secret|token|password|credential|access[_-]?key|authorization)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(r"(^|\b)(sk-|sk_|rk-|AKIA|ASIA|ABSK)[A-Za-z0-9_\-]{8,}")
_REGION_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d$")
_PROFILE_RE = re.compile(r"^[A-Za-z0-9_.+=,@\-]{1,128}$")
_URL_RE = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d+)?(/[A-Za-z0-9._~/\-]*)?$")


class RuntimeConfigError(ValueError):
    """One or more configuration problems, reported before the match starts."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass(frozen=True)
class AgentRuntimeConfig:
    policy: str
    personality: str = "cautious"
    provider: str | None = None
    model: str | None = None
    identity: str | None = None           # surrogate identity (sonnet55 | gpt61sol); default from the model
    provider_options: dict[str, Any] = field(default_factory=dict)
    max_provider_calls: int | None = None  # hybrid: the escalation budget of this spy (provider calls)

    @property
    def descriptor(self) -> pr.PolicyDescriptor:
        return pr.POLICIES[self.policy]

    @property
    def calls_provider(self) -> bool:
        return self.policy in pr.POLICIES and pr.POLICIES[self.policy].requires_provider

    @property
    def surrogate_identity(self) -> str:
        return pr.identity_for(self.policy, self.provider, self.model, self.identity)

    @property
    def policy_key(self) -> str:
        return pr.policy_key(self.surrogate_identity, self.personality)

    def public(self) -> dict[str, Any]:
        out = {"policy": self.policy, "personality": self.personality}
        if self.calls_provider:
            out.update({"provider": self.provider, "model": self.model})
            if self.provider_options:
                out["provider_options"] = dict(self.provider_options)
            if self.max_provider_calls is not None:
                out["max_provider_calls"] = self.max_provider_calls
        if self.identity is not None:
            out["identity"] = self.identity
        return out

    def effective(self) -> dict[str, Any]:
        """public() plus what the runtime derives (the surrogate identity / policy key it plays)."""
        out = self.public()
        if self.descriptor.kind in ("surrogate", "hybrid"):
            out.update({"surrogate_identity": self.surrogate_identity, "policy_key": self.policy_key})
        return out


@dataclass(frozen=True)
class GameRuntimeConfig:
    black: AgentRuntimeConfig
    white: AgentRuntimeConfig
    level: str = "level_01"
    seed: int = 42
    mode: str = "playable"
    view: str = "play"
    speed: float = 1.0
    show_telemetry: bool = True
    time_limit: float = 300.0
    save_replay: bool = True
    output_dir: str = DEFAULT_OUTPUT_DIR
    # door traps, trap variants and room dressing (playable runtime rules, spec 35); None = auto: on in playable
    # mode, off in evaluation mode. The resolved value is what public() records (a replay replays exactly that).
    enhanced_gameplay: bool | None = None

    def agent(self, side: str) -> AgentRuntimeConfig:
        return {"black": self.black, "white": self.white, "black_spy": self.black, "white_spy": self.white}[side]

    @property
    def providers_used(self) -> tuple[str, ...]:
        return tuple(sorted({a.provider for a in (self.black, self.white) if a.calls_provider and a.provider}))

    @property
    def has_hybrid(self) -> bool:
        return any(a.policy in pr.POLICIES and a.descriptor.kind == "hybrid" for a in (self.black, self.white))

    def public(self) -> dict[str, Any]:
        """Everything needed to rerun the configuration, nothing secret (provider options never hold secrets)."""
        return {"black": self.black.public(), "white": self.white.public(), "level": self.level, "seed": self.seed,
                "mode": self.mode, "view": self.view, "speed": self.speed, "show_telemetry": self.show_telemetry,
                "time_limit": self.time_limit, "save_replay": self.save_replay, "output_dir": self.output_dir,
                "enhanced_gameplay": enhanced(self)}

    def with_agent(self, side: str, **changes: Any) -> "GameRuntimeConfig":
        return replace(self, **{side: replace(self.agent(side), **changes)})


# -- execution semantics (spec 20) -------------------------------------------------------------------------

EVALUATION_LABEL = "EVALUATION MODE - frozen research execution semantics (a runtime match is not a scientific result)"
PLAYABLE_LABEL = "PLAYABLE MODE - execution semantics differ from the frozen evaluation protocol"
ENHANCED_LABEL = "Enhanced gameplay - trap target/type resolution occurs downstream of the frozen policy"
ENHANCED_REASON = ("enhanced gameplay: door traps, trap variants and room dressing are playable-runtime rules "
                   "(never part of Stages 3-5)")
ENHANCED_CHOICES = {"auto": None, "on": True, "off": False}


def enhanced(config: GameRuntimeConfig) -> bool:
    """Whether the enhanced gameplay rules run: the explicit setting, else on in playable mode only."""
    if config.enhanced_gameplay is None:
        return config.mode == "playable"
    return bool(config.enhanced_gameplay)


def execution(config: GameRuntimeConfig) -> dict[str, Any]:
    """The live-game settings a mode stands for. Evaluation reproduces the Stage 3-5 run settings
    (workshop_runs/stage5c-confirmatory/matches/*/run.json): evaluation timing, simulation_time + game_time,
    barrier_seeded_order, native runtime, Bedrock temperature provider-default, 120 s timeout."""
    on = enhanced(config)
    common = {"runtime": "native", "mission_clock": "simulation_time", "encounters": True, "lives": True,
              "items": "contested", "puzzle": "riddle", "spawn": "different_rooms", "time_limit": config.time_limit,
              "enhanced_gameplay": on}
    if config.mode == "evaluation":
        return {**common, "decision_timing": "evaluation", "mission_timer": "game_time",
                "commit_policy": "barrier_seeded_order", "label": EVALUATION_LABEL, "notes": [ENHANCED_LABEL] if on else []}
    notes = [ENHANCED_LABEL] if on else []
    timing, timer = "live_async", "wall_clock"
    if config.has_hybrid:
        # The HybridController is validated only with evaluation timing, whose contract is that provider latency spends
        # no mission time (workshop/live/config.py refuses hybrids otherwise). The world freezes while a model thinks,
        # so the mission clock must freeze with it: on a wall-clock timer the clock drains during the freeze, every
        # move then looks stalled to the action lifecycle (no progress for 1 s of mission time) and both spies end
        # up re-deciding every frame without moving.
        timing, timer = "evaluation", "game_time"
        notes.append("a hybrid spy runs with evaluation decision timing (the only timing the HybridController was "
                     "validated with): the world and the mission clock pause while a model call is in flight")
    return {**common, "decision_timing": timing, "mission_timer": timer, "commit_policy": "barrier_answer_order",
            "label": PLAYABLE_LABEL, "notes": notes}


# -- validation ---------------------------------------------------------------------------------------------

def _secret_like(value: Any) -> bool:
    return isinstance(value, str) and bool(_SECRET_VALUE_RE.search(value))


def _agent_problems(side: str, a: AgentRuntimeConfig) -> list[str]:
    who = f"{side.capitalize()} Spy"
    problems: list[str] = []
    if a.policy not in pr.POLICIES:
        return [f"{who}: unknown policy '{a.policy}' (choose one of {', '.join(pr.POLICY_IDS)})"]
    d = a.descriptor
    if a.personality not in pr.PERSONALITIES:
        problems.append(f"{who}: unknown personality '{a.personality}' (choose one of {', '.join(pr.PERSONALITIES)})")
    if a.identity is not None and a.identity not in pr.IDENTITIES:
        problems.append(f"{who}: unknown surrogate identity '{a.identity}' (choose one of {', '.join(pr.IDENTITIES)})")
    if a.identity is not None and d.kind not in ("surrogate", "hybrid"):
        problems.append(f"{who}: {d.label} has no surrogate, so it takes no identity")
    if not d.requires_provider:
        if a.provider or a.model or a.provider_options or a.max_provider_calls is not None:
            problems.append(f"{who}: {d.label} makes no provider call - remove the provider / model settings")
        return problems
    if not a.provider:
        problems.append(f"{who}: {d.label} needs a provider ({', '.join(PROVIDER_IDS)})")
    elif a.provider not in PROVIDERS:
        problems.append(f"{who}: unknown provider '{a.provider}' (choose one of {', '.join(PROVIDER_IDS)})")
    if not a.model:
        problems.append(f"{who}: {d.label} needs a model id for its provider")
    elif _secret_like(a.model):
        problems.append(f"{who}: the model value looks like a credential - never pass secrets as settings")
    elif "<" in a.model or not _MODEL_RE.match(a.model):
        problems.append(f"{who}: '{a.model}' is not a valid model identifier")
    allowed = PROVIDERS[a.provider].options if a.provider in PROVIDERS else ()
    for key, value in (a.provider_options or {}).items():
        if _SECRET_KEY_RE.search(str(key)) or _secret_like(value):
            problems.append(f"{who}: provider option '{key}' looks like a secret - credentials come only from the "
                            "environment / AWS credential chain, never from the configuration")
        elif key not in allowed:
            problems.append(f"{who}: unknown {a.provider} option '{key}' (allowed: {', '.join(allowed) or 'none'})")
    opts = a.provider_options or {}
    if "region" in opts and not _REGION_RE.match(str(opts["region"])):
        problems.append(f"{who}: '{opts['region']}' is not an AWS region")
    if "aws_profile" in opts and not _PROFILE_RE.match(str(opts["aws_profile"])):
        problems.append(f"{who}: '{opts['aws_profile']}' is not a valid AWS profile name")
    if "base_url" in opts and not _URL_RE.match(str(opts["base_url"])):
        problems.append(f"{who}: '{opts['base_url']}' is not an http(s) URL")
    if "timeout_s" in opts:
        try:
            if not 1 <= float(opts["timeout_s"]) <= 600:
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"{who}: timeout_s must be between 1 and 600 seconds")
    if "temperature" in opts and str(opts["temperature"]) not in ("0", "0.0", "provider-default"):
        problems.append(f"{who}: temperature must be 0 or provider-default")
    if a.max_provider_calls is not None:
        if d.kind != "hybrid":
            problems.append(f"{who}: max_provider_calls is the hybrid escalation budget (not used by {d.label})")
        elif not isinstance(a.max_provider_calls, int) or a.max_provider_calls < 0:
            problems.append(f"{who}: max_provider_calls must be a non-negative integer")
    return problems


def problems(config: GameRuntimeConfig) -> list[str]:
    out = _agent_problems("black", config.black) + _agent_problems("white", config.white)
    if config.level not in LEVELS:
        out.append(f"unknown level '{config.level}' (choose one of {', '.join(LEVELS)})")
    if not isinstance(config.seed, int) or isinstance(config.seed, bool) or not 0 <= config.seed < 2 ** 31:
        out.append("seed must be an integer between 0 and 2^31-1")
    if config.mode not in MODES:
        out.append(f"unknown mode '{config.mode}' (choose one of {', '.join(MODES)})")
    if config.view not in VIEWS:
        out.append(f"unknown view '{config.view}' (choose one of {', '.join(VIEWS)})")
    if config.speed not in SPEEDS:
        out.append(f"speed must be one of {', '.join(f'{s:g}' for s in SPEEDS)}")
    if config.enhanced_gameplay is not None and not isinstance(config.enhanced_gameplay, bool):
        out.append("enhanced_gameplay must be true, false or auto (unset)")
    lo, hi = TIME_LIMIT_RANGE
    if not isinstance(config.time_limit, int | float) or not lo <= float(config.time_limit) <= hi:
        out.append(f"time_limit must be between {lo:g} and {hi:g} seconds")
    out_dir = Path(config.output_dir)
    parts = set(out_dir.resolve().parts) if out_dir.is_absolute() else set(out_dir.parts)
    if {"policies", "policy_data", "policy_reports"} & parts or "workshop_runs" in parts:
        out.append("output_dir must not be inside the research archive (workshop/policies, policy_data, "
                   "policy_reports) or workshop_runs/")
    return out


def validate(config: GameRuntimeConfig) -> GameRuntimeConfig:
    found = problems(config)
    if found:
        raise RuntimeConfigError(found)
    return config


# -- loading (YAML / JSON / dict / CLI) ----------------------------------------------------------------------

_AGENT_KEYS = {"policy", "personality", "provider", "model", "identity", "provider_options", "options", "max_provider_calls"}
_GAME_KEYS = {"black", "white", "level", "seed", "mode", "view", "speed", "show_telemetry", "time_limit", "save_replay",
              "output_dir", "enhanced_gameplay"}


def _agent_from(side: str, data: Any) -> AgentRuntimeConfig:
    if not isinstance(data, dict):
        raise RuntimeConfigError([f"'{side}' must be a mapping of policy / provider / model / personality"])
    unknown = sorted(set(data) - _AGENT_KEYS)
    if unknown:
        raise RuntimeConfigError([f"{side}: unknown key(s) {', '.join(unknown)} (known: {', '.join(sorted(_AGENT_KEYS))})"])
    for key in data:
        if _SECRET_KEY_RE.search(str(key)):
            raise RuntimeConfigError([f"{side}: '{key}' looks like a secret - never put credentials in a configuration"])
    options = data.get("provider_options", data.get("options")) or {}
    if not isinstance(options, dict):
        raise RuntimeConfigError([f"{side}: provider_options must be a mapping"])
    calls = data.get("max_provider_calls")
    return AgentRuntimeConfig(policy=str(data.get("policy") or ""), personality=str(data.get("personality") or "cautious"),
                              provider=data.get("provider") or None, model=None if data.get("model") is None else str(data["model"]),
                              identity=data.get("identity") or None, provider_options=dict(options),
                              max_provider_calls=int(calls) if isinstance(calls, int | float) and not isinstance(calls, bool)
                              else calls)


def from_dict(data: dict[str, Any]) -> GameRuntimeConfig:
    if not isinstance(data, dict):
        raise RuntimeConfigError(["a configuration is a mapping with black / white sections"])
    unknown = sorted(set(data) - _GAME_KEYS)
    if unknown:
        raise RuntimeConfigError([f"unknown key(s) {', '.join(unknown)} (known: {', '.join(sorted(_GAME_KEYS))})"])
    for side in ACTORS:
        if side not in data:
            raise RuntimeConfigError([f"missing '{side}' section"])
    kwargs: dict[str, Any] = {k: data[k] for k in _GAME_KEYS - {"black", "white"} if k in data and data[k] is not None}
    if "speed" in kwargs:
        kwargs["speed"] = float(kwargs["speed"])
    if "time_limit" in kwargs:
        kwargs["time_limit"] = float(kwargs["time_limit"])
    if isinstance(kwargs.get("enhanced_gameplay"), str) and kwargs["enhanced_gameplay"].lower() in ENHANCED_CHOICES:
        kwargs["enhanced_gameplay"] = ENHANCED_CHOICES[kwargs["enhanced_gameplay"].lower()]
    return GameRuntimeConfig(black=_agent_from("black", data["black"]), white=_agent_from("white", data["white"]), **kwargs)


def parse_scalar(text: str) -> Any:
    t = text.strip()
    if t == "" or t in ("~", "null", "None"):
        return None
    if t in ("true", "True", "yes", "on"):
        return True
    if t in ("false", "False", "no", "off"):
        return False
    if (t[0] == t[-1]) and t[0] in "'\"" and len(t) >= 2:
        return t[1:-1]
    if t.startswith("{") or t.startswith("["):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def load_yaml_subset(text: str) -> dict[str, Any]:
    """The YAML the runtime configs use (nested mappings, scalars, comments), when PyYAML is not installed."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, rest = line.strip().partition(":")
        if not sep:
            raise RuntimeConfigError([f"line {number}: expected 'key: value'"])
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest.strip() == "":
            parent[key.strip()] = {}
            stack.append((indent, parent[key.strip()]))
        else:
            parent[key.strip()] = parse_scalar(rest)
    return root


def load_file(path: str | Path) -> GameRuntimeConfig:
    p = Path(path).expanduser()
    if not p.is_file():
        raise RuntimeConfigError([f"config file not found: {p}"])
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # PyYAML, when installed
            data = yaml.safe_load(text)
        except ImportError:
            data = load_yaml_subset(text)
    return from_dict(data or {})


def to_yaml(config: GameRuntimeConfig) -> str:
    """A configuration file for this config (the setup screen's save; no secret can be in it)."""
    def emit(data: dict[str, Any], indent: int = 0) -> list[str]:
        lines = []
        for k, v in data.items():
            if isinstance(v, dict):
                lines.append(" " * indent + f"{k}:")
                lines += emit(v, indent + 2)
            else:
                lines.append(" " * indent + f"{k}: {json.dumps(v) if isinstance(v, str) and (':' in v or '#' in v) else v}")
        return lines
    body = config.public()
    if config.enhanced_gameplay is None:
        body["enhanced_gameplay"] = "auto"  # a saved configuration keeps "auto" (on in playable, off in evaluation)
    for side in ACTORS:
        body[side] = {k: v for k, v in body[side].items()}
    return "\n".join(emit(body)) + "\n"


def describe(config: GameRuntimeConfig) -> list[str]:
    """Human-readable lines: each spy, its scientific status and its compatibility label."""
    lines = []
    ex = execution(config)
    for side in ACTORS:
        a = config.agent(side)
        d = a.descriptor
        comp = pr.compatibility(a.policy, provider=a.provider, model=a.model, identity=a.surrogate_identity,
                                level=config.level, mode=config.mode, time_limit=config.time_limit,
                                enhanced=enhanced(config))
        where = f" on {a.provider}/{a.model}" if a.calls_provider else ""
        ident = f" (surrogate {a.policy_key})" if d.kind in ("surrogate", "hybrid") else ""
        lines.append(f"{side.capitalize()} Spy: {d.label}{where}{ident}, personality {a.personality}")
        lines.append(f"    status: {d.scientific_label}" + (f" ({d.summary[0]})" if d.kind == "hybrid" and d.id == "hybrid_v2" else ""))
        lines.append(f"    {comp.label}: {'; '.join(comp.reasons)}")
    lines.append(f"{config.level}, seed {config.seed}, {config.time_limit:g} s, speed {config.speed:g}x, view {config.view}")
    lines.append(ex["label"])
    lines += [f"    note: {n}" for n in ex["notes"]]
    return lines


__all__ = ["AgentRuntimeConfig", "GameRuntimeConfig", "RuntimeConfigError", "validate", "problems", "execution",
           "from_dict", "load_file", "to_yaml", "describe", "enhanced", "LEVELS", "MODES", "VIEWS", "SPEEDS", "asdict",
           "ENHANCED_LABEL", "ENHANCED_REASON", "ENHANCED_CHOICES"]
