"""Live-play configuration: provider, model and runtime per spy.

Sources, lowest to highest precedence for the same key:

    1. a TOML file (``--config``, see workshop/live_config.example.toml)
    2. the command line

Then, per spy, an actor-specific key (``black_model``) always beats the
shared key (``model``). Nothing is guessed: there is no default provider and
no default model, an unknown provider is an error (never a silent mock), and
secrets are never accepted here - the OpenAI key comes only from the
``OPENAI_API_KEY`` environment variable and AWS credentials only from the
normal AWS credential chain or ``AWS_BEARER_TOKEN_BEDROCK``.
"""
from __future__ import annotations

import json

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import constants
from src.agents.live_async import DECISION_TIMING_MODES, LIVE_ASYNC
from src.mission_clock import MISSION_CLOCK_MODES, MISSION_TIMERS, SIMULATION_TIME, WALL_CLOCK
from src.llm.personality import VALID_PERSONALITY_PROFILES, InvalidPersonalityProfileError, validate_personality_profile

SUPPORTED_PROVIDERS = ("bedrock", "openai")
# Who plays a spy (Stage 3): a real LLM agent (the default, everything below), a trained
# policy-emulator artifact (workshop/policy_emulator, no provider call) or the engine's
# scripted bot. A non-LLM spy is never a silent fallback: it is chosen explicitly per spy.
CONTROLLERS = ("llm", "surrogate", "scripted", "hybrid")
# Stage 3 milestone 4: a hybrid spy plays its B2 surrogate and hands selected decisions to the real
# LLM of the same identity (workshop/policy_emulator/hybrid.py, routers in workshop/policy_emulator/router.py)
HYBRID_ROUTERS = ("R0", "R1", "R2", "R3", "R2.1")  # R2.1: hybrid-router-v2 (milestone 4C)
HYBRID_BUDGET_KEYS = ("max_llm_calls_per_spy", "max_llm_calls_per_match", "max_llm_calls_total")
DEFAULT_CONTROLLER = "llm"
POLICY_EMULATORS = {"table": "table-v1", "frequency": "frequency-v1", "logistic": "logistic-v1", "tree": "tree-v1"}
POLICY_TARGETS = {"v1": "target-v1", "v2": "target-v2", "v2.1": "target-v2.1"}
DEFAULT_POLICY_TARGET = "target-v2"
DEFAULT_POLICY_DATASET = "phase-a-v2"
SCRIPTED_MODEL = "scripted_active_v1"
# Workshop runtime name -> agent-runtime registry name. "native" is the
# production Native runtime as validated in M6G.3 (the tool-like
# representation over the plain completion API); "strands" is the Strands
# adapter. Both keep the same Observation -> action -> validation/repair/
# fallback -> scheduler -> Engine boundary.
RUNTIMES = {"native": "native_tool_like", "strands": "strands"}
DEFAULT_RUNTIME = "native"
# OpenAI: 0 matches every other provider path in this project (BedrockClient
# and OpenAICompatibleClient both send temperature 0). Some OpenAI models
# only accept their own default; "provider-default" omits the field.
OPENAI_TEMPERATURES = ("0", "provider-default")
# Bedrock: the same choice. Newer Claude models on Bedrock (Sonnet 5 / 5.5,
# Opus 4.8 / 5 / 5.5 - preflight 2026-10-01) reject `temperature` outright
# ("temperature is deprecated for this model"); "provider-default" omits it.
BEDROCK_TEMPERATURES = ("0", "provider-default")
DEFAULT_TIMEOUT_SECONDS = 60.0
OPENAI_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"

ACTORS = (constants.BLACK_SPY_ID, constants.WHITE_SPY_ID)
ACTOR_LABELS = {constants.BLACK_SPY_ID: "Black Spy", constants.WHITE_SPY_ID: "White Spy"}
ACTOR_PREFIX = {constants.BLACK_SPY_ID: "black", constants.WHITE_SPY_ID: "white"}

# Decision timing (bugfix live flow, Part A). The workshop plays in
# live_async: the game clock runs while a model thinks, each spy commits on
# its own, and a stale answer is re-checked against the world before it is
# applied (src/agents/live_async.py). "evaluation" is the frozen-clock M6G.3
# barrier every experiment and stored replay uses.
DEFAULT_DECISION_TIMING = LIVE_ASYNC
PERSONALITIES = tuple(sorted(VALID_PERSONALITY_PROFILES))
# Personality presets (bugfix live flow, Part C). Names read Black-vs-White.
# They only select existing personality profiles (src/llm/personality.py) -
# no scripted behavior, no extra prompt text.
PERSONALITY_PRESETS: dict[str, tuple[str, str]] = {
    "balanced": ("default", "default"),
    "racer-vs-saboteur": ("racer", "saboteur"),
    "saboteur-vs-racer": ("saboteur", "racer"),
    "saboteur-vs-saboteur": ("saboteur", "saboteur"),
    "rogue-vs-cautious": ("saboteur", "cautious"),  # V2 Part D: sabotage + approach vs avoidance
    "rogue-vs-rogue": ("saboteur", "saboteur"),  # both sabotage (same profiles as saboteur-vs-saboteur)
    "cautious-vs-opportunist": ("cautious", "opportunist"),
}
DEFAULT_PERSONALITY_PRESET = "balanced"

# Mission time budget V2 (§9-10, §43): the live mission length, one central
# table. The engine default (constants.MISSION_TIME, 180 s) stays what every
# recorded run used; the live workshop default is "standard", 5 minutes.
MISSION_PRESETS: dict[str, float] = {"quick": 150.0, "standard": 300.0, "long": 480.0}
DEFAULT_MISSION_PRESET = "standard"
# action_budget: each spy's mission time does not run while it only waits for
# its own model (provider latency is not mission skill); simulation_time: the
# one shared clock runs regardless (src/mission_clock.py).
# Live feedback: the mission timer must not stop while the spies think, so the
# workshop default is the one shared, always-running clock; action_budget
# (per-spy time that pauses while its own model thinks) stays selectable.
DEFAULT_MISSION_CLOCK = SIMULATION_TIME
# V3: with the shared clock, what drives the countdown - real (unpaused) wall
# time by default, so a slow frame or a thinking model never slows it;
# game_time = the world's capped frame dt (the earlier behaviour).
DEFAULT_MISSION_TIMER = WALL_CLOCK
# live commit policies (src/game.py): the default barrier, and the seeded order of the cross-model lab
COMMIT_POLICY_CHOICES = ("barrier_answer_order", "barrier_seeded_order")
TIME_LIMIT_RANGE = (30.0, 1800.0)

# Keys accepted in the TOML file (and, with dashes, on the command line).
CONFIG_KEYS = (
    "provider", "model", "runtime", "region", "aws_profile", "timeout_seconds", "openai_temperature", "bedrock_temperature", "seed",
    "black_provider", "black_model", "white_provider", "white_model",
    "request_fields", "black_request_fields", "white_request_fields",
    "decision_timing", "personality_preset", "black_personality", "white_personality", "encounters",
    "time_limit", "mission_preset", "mission_clock", "mission_timer", "lives", "spawn", "items", "puzzle",
    "commit_policy",
    "black_controller", "white_controller", "black_policy", "white_policy", "policy_dataset", "policy_emulator",
    "surrogate_seed", "policy_target",
    "hybrid_router", "max_llm_calls_per_spy", "max_llm_calls_per_match", "max_llm_calls_total",
)
_SECRET_KEY_RE = re.compile(r"(api[_-]?key|secret|token|password|credential|access[_-]?key)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(r"^(sk-|sk_|AKIA|ASIA|ABSK)[A-Za-z0-9_\-]{8,}")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_PROFILE_RE = re.compile(r"^[A-Za-z0-9_.+=,@\-]{1,128}$")
_REGION_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d$")


class LiveConfigError(ValueError):
    """A live-play configuration problem, reported before anything starts."""


@dataclass(frozen=True)
class ActorSpec:
    actor_id: str
    provider: str
    model: str
    personality: str = "default"
    # Bedrock only: extra model request fields (Converse additionalModelRequestFields), canonical JSON.
    # Found 2026-10-02: GPT-6.1 Sol on Bedrock answers a game request in ~33 s at its default
    # reasoning effort and in ~2 s with {"reasoning": {"effort": "low"}}.
    request_fields: str | None = None
    # Stage 3: llm (default) / surrogate (provider "surrogate", model = the policy key) / scripted
    controller: str = DEFAULT_CONTROLLER
    policy: str | None = None  # surrogate only: <model>:<personality>, e.g. gpt61sol:opportunist

    @property
    def is_llm(self) -> bool:
        return self.controller == DEFAULT_CONTROLLER
    @property
    def calls_provider(self) -> bool:
        """An LLM spy, or a hybrid spy (its escalated decisions go to the real model)."""
        return self.controller in (DEFAULT_CONTROLLER, "hybrid")

    @property
    def label(self) -> str:
        return ACTOR_LABELS[self.actor_id]

    def target(self) -> tuple[str, str] | tuple[str, str, str]:
        return (self.provider, self.model) if not self.request_fields else (self.provider, self.model, self.request_fields)


@dataclass(frozen=True)
class LiveConfig:
    black: ActorSpec
    white: ActorSpec
    runtime: str = DEFAULT_RUNTIME
    region: str | None = None
    aws_profile: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    openai_temperature: str = "0"
    bedrock_temperature: str = "0"
    seed: int | None = None
    decision_timing: str = DEFAULT_DECISION_TIMING
    personality_preset: str | None = DEFAULT_PERSONALITY_PRESET
    # V2 Part D: engine-owned spy fights when the spies touch (src/encounters.py)
    encounters: bool = True
    time_limit: float = MISSION_PRESETS[DEFAULT_MISSION_PRESET]
    mission_preset: str | None = DEFAULT_MISSION_PRESET
    mission_clock: str = DEFAULT_MISSION_CLOCK
    mission_timer: str = DEFAULT_MISSION_TIMER
    # the M6G.3 barrier order (evaluation timing): first-ready (default) or seeded,
    # so a faster model cannot win same-tick races by answering sooner
    commit_policy: str = "barrier_answer_order"
    lives: bool = True  # 3 lives: each trap that goes off under a spy costs one
    spawn: str = "different_rooms"  # src/spawns.py: different_rooms (default) or same_room
    items: str = "contested"  # contested: one copy of each required item; shared: each spy finds its own
    puzzle: str = "riddle"  # the safe: riddle (work the code out), split (two halves), none (one code object)
    # Stage 3 surrogate spies: which trained artifacts, and the sampling seed (default: the match seed)
    policy_dataset: str = DEFAULT_POLICY_DATASET
    policy_emulator: str = POLICY_EMULATORS["table"]
    policy_target: str = DEFAULT_POLICY_TARGET
    surrogate_seed: int | None = None
    # Stage 3 milestone 4: the router of every hybrid spy, and the provider-call caps (None = no cap)
    hybrid_router: str | None = None
    max_llm_calls_per_spy: int | None = None
    max_llm_calls_per_match: int | None = None
    max_llm_calls_total: int | None = None
    sources: dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def actors(self) -> dict[str, ActorSpec]:
        return {constants.BLACK_SPY_ID: self.black, constants.WHITE_SPY_ID: self.white}

    @property
    def registry_runtime(self) -> str:
        return RUNTIMES[self.runtime]

    @property
    def providers_used(self) -> tuple[str, ...]:
        """The LLM providers this match calls (a surrogate or scripted spy calls none)."""
        return tuple(sorted({s.provider for s in (self.black, self.white) if s.calls_provider}))

    @property
    def llm_actors(self) -> tuple[str, ...]:
        return tuple(a for a, s in self.actors.items() if s.calls_provider)
    @property
    def hybrid_actors(self) -> tuple[str, ...]:
        return tuple(a for a, s in self.actors.items() if s.controller == "hybrid")

    @property
    def non_llm(self) -> bool:
        return any(not s.is_llm for s in (self.black, self.white))

    @property
    def mixed(self) -> bool:
        return self.black.target() != self.white.target()

    def openai_temperature_value(self) -> float | None:
        return None if self.openai_temperature == "provider-default" else 0.0

    def bedrock_temperature_value(self) -> float | None:
        return None if self.bedrock_temperature == "provider-default" else 0.0

    def to_public_dict(self) -> dict[str, Any]:
        """Everything needed to reproduce the configuration - and nothing secret."""
        def actor(spec: ActorSpec) -> dict[str, Any]:
            return {"provider": spec.provider, "model": spec.model, "personality": spec.personality,
                    **({"request_fields": json.loads(spec.request_fields)} if spec.request_fields else {}),
                    # Stage 3: only a non-LLM spy carries these keys (an LLM-only configuration is unchanged)
                    **({"controller": spec.controller, "policy": spec.policy} if not spec.is_llm else {})}
        surrogate = any(s.controller in ("surrogate", "hybrid") for s in (self.black, self.white))
        hybrid = any(s.controller == "hybrid" for s in (self.black, self.white))
        return {
            "black": actor(self.black),
            "white": actor(self.white),
            **({"policy_dataset": self.policy_dataset, "policy_emulator": self.policy_emulator,
                "policy_target": self.policy_target, "surrogate_seed": self.surrogate_seed} if surrogate else {}),
            **({"hybrid_router": self.hybrid_router, **{k: getattr(self, k) for k in HYBRID_BUDGET_KEYS}} if hybrid else {}),
            "decision_timing": self.decision_timing,
            "personality_preset": self.personality_preset,
            "encounters": self.encounters,
            "time_limit_s": self.time_limit,
            "mission_preset": self.mission_preset,
            "mission_clock": self.mission_clock,
            "mission_timer": self.mission_timer,
            "commit_policy": self.commit_policy,
            "lives": self.lives,
            "spawn": self.spawn,
            "items": self.items,
            "puzzle": self.puzzle,
            "runtime": self.runtime,
            "agent_runtime": self.registry_runtime,
            "region": self.region if "bedrock" in self.providers_used else None,
            "aws_profile": self.aws_profile if "bedrock" in self.providers_used else None,
            "timeout_seconds": self.timeout_seconds,
            "openai_temperature": self.openai_temperature if "openai" in self.providers_used else None,
            "bedrock_temperature": self.bedrock_temperature if "bedrock" in self.providers_used else None,
            "seed": self.seed,
            "mixed_configuration": self.mixed,
            "sources": dict(self.sources),
        }


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Reads a live-play TOML file: top-level keys only, known keys only, no secrets."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise LiveConfigError(f"config file not found: {p}")
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise LiveConfigError(f"{p} is not valid TOML: {exc}") from None
    # An optional [live] table is accepted as an alias for the top level.
    if set(data) == {"live"} and isinstance(data["live"], dict):
        data = data["live"]
    for key, value in data.items():
        if _SECRET_KEY_RE.search(key):
            raise LiveConfigError(
                f"{p}: '{key}' looks like a secret. Secrets are never read from config files - "
                f"set {OPENAI_KEY_ENV} in the environment (OpenAI) or use your AWS profile/credential chain (Bedrock)."
            )
        if key not in CONFIG_KEYS:
            raise LiveConfigError(f"{p}: unknown key '{key}'. Known keys: {', '.join(CONFIG_KEYS)}")
        if isinstance(value, dict | list):
            raise LiveConfigError(f"{p}: '{key}' must be a single value, not a table or list")
    return {k: v for k, v in data.items() if v not in (None, "")}


def load_config_files(paths) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Several TOML files, merged in order - a later file wins per key (the
    command line still wins over all of them). Typical split: one file for
    the characters' behaviour (personalities, encounters, lives, mission) and
    one for the models (provider, model, runtime, region).
    Returns (values, key -> file name it came from, override notes)."""
    merged: dict[str, Any] = {}
    origin: dict[str, str] = {}
    notes: list[str] = []
    for path in paths or ():
        name = Path(path).name
        for key, value in load_config_file(path).items():
            if key in merged and merged[key] != value:
                notes.append(f"{key}: {name} overrides {origin[key]} ({merged[key]!r} -> {value!r})")
            merged[key] = value
            origin[key] = name
    return merged, origin, notes


def _clean_str(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LiveConfigError(f"'{key}' must be a string, got {type(value).__name__}")
    value = value.strip()
    return value or None


def _validate_model(model: str, who: str, provider: str) -> str:
    if "<" in model or ">" in model:
        raise LiveConfigError(f"{who}: model '{model}' is still a placeholder - replace it with a real {provider} model id")
    if _SECRET_VALUE_RE.match(model):
        raise LiveConfigError(f"{who}: the model value looks like a credential, not a model id. Never pass secrets as settings.")
    if not _MODEL_RE.match(model):
        raise LiveConfigError(f"{who}: '{model}' is not a valid model identifier")
    return model


def resolve_config(cli: dict[str, Any], file_values: dict[str, Any] | None = None,
                   file_sources: dict[str, str] | None = None) -> LiveConfig:
    """Merges file and command-line values (CLI wins per key) and validates the result.
    `file_sources` (from load_config_files) names the file each file key came from."""
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for origin, values in (("config file", file_values or {}), ("command line", cli)):
        for key, value in values.items():
            if key in CONFIG_KEYS and value not in (None, ""):
                merged[key] = value
                sources[key] = (f"config file {file_sources[key]}" if origin == "config file" and file_sources
                                and key in file_sources else origin)

    preset, personalities = _resolve_personalities(merged)
    specs = {}
    for actor_id in ACTORS:
        prefix = ACTOR_PREFIX[actor_id]
        label = ACTOR_LABELS[actor_id]
        controller = (_clean_str(merged.get(f"{prefix}_controller"), f"{prefix}_controller") or DEFAULT_CONTROLLER).lower()
        if controller not in CONTROLLERS:
            raise LiveConfigError(f"{label}: unknown controller '{controller}'. Available: {', '.join(CONTROLLERS)}")
        if controller == "hybrid":
            specs[actor_id] = _hybrid_spec(actor_id, merged)
            continue
        if controller != DEFAULT_CONTROLLER:
            specs[actor_id] = _non_llm_spec(actor_id, controller, merged, personalities[actor_id])
            continue
        provider = _clean_str(merged.get(f"{prefix}_provider") or merged.get("provider"), "provider")
        model = _clean_str(merged.get(f"{prefix}_model") or merged.get("model"), "model")
        if provider is None:
            raise LiveConfigError(
                f"no provider for {label}: pass --provider bedrock|openai (both spies) or --{prefix}-provider. "
                "The live game never picks a provider for you."
            )
        provider = provider.lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise LiveConfigError(
                f"{label}: unknown provider '{provider}'. Supported for live play: {', '.join(SUPPORTED_PROVIDERS)}. "
                "There is no fallback to a mock or scripted agent."
            )
        if model is None:
            raise LiveConfigError(
                f"no model for {label} ({provider}): pass --model <id> (both spies) or --{prefix}-model <id>. "
                "The workshop does not hard-code a model."
            )
        specs[actor_id] = ActorSpec(actor_id=actor_id, provider=provider, model=_validate_model(model, label, provider),
                                    personality=personalities[actor_id],
                                    request_fields=_request_fields(merged.get(f"{prefix}_request_fields")
                                                                   or merged.get("request_fields"), label, provider))

    runtime = (_clean_str(merged.get("runtime"), "runtime") or DEFAULT_RUNTIME).lower()
    if runtime not in RUNTIMES:
        raise LiveConfigError(f"unknown runtime '{runtime}'. Available: {', '.join(RUNTIMES)}")
    if runtime == "strands":
        for spec in specs.values():
            if spec.is_llm and spec.provider != "bedrock":
                raise LiveConfigError(
                    f"{spec.label}: the Strands runtime supports only the bedrock provider in this project "
                    f"(its adapter has no OpenAI model). Use --runtime native for {spec.provider}."
                )

    region = _clean_str(merged.get("region"), "region")
    if region is not None and not _REGION_RE.match(region):
        raise LiveConfigError(f"'{region}' is not an AWS region name (e.g. us-east-1)")
    aws_profile = _clean_str(merged.get("aws_profile"), "aws_profile")
    if aws_profile is not None and not _PROFILE_RE.match(aws_profile):
        raise LiveConfigError(f"'{aws_profile}' is not a valid AWS profile name")

    timeout = merged.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        raise LiveConfigError(f"timeout_seconds must be a number, got {timeout!r}") from None
    if not 1.0 <= timeout <= 600.0:
        raise LiveConfigError("timeout_seconds must be between 1 and 600")

    temperature = str(merged.get("openai_temperature", "0")).strip().lower()
    if temperature in ("0.0", "0"):
        temperature = "0"
    if temperature not in OPENAI_TEMPERATURES:
        raise LiveConfigError(f"openai_temperature must be one of {', '.join(OPENAI_TEMPERATURES)}")
    bedrock_temperature = str(merged.get("bedrock_temperature", "0")).strip().lower()
    if bedrock_temperature in ("0.0", "0"):
        bedrock_temperature = "0"
    if bedrock_temperature not in BEDROCK_TEMPERATURES:
        raise LiveConfigError(f"bedrock_temperature must be one of {', '.join(BEDROCK_TEMPERATURES)}")

    seed = merged.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int | str):
            raise LiveConfigError("seed must be an integer")
        try:
            seed = int(seed)
        except ValueError:
            raise LiveConfigError(f"seed must be an integer, got {seed!r}") from None

    timing = (_clean_str(merged.get("decision_timing"), "decision_timing") or DEFAULT_DECISION_TIMING).lower().replace("-", "_")
    if timing not in DECISION_TIMING_MODES:
        raise LiveConfigError(f"unknown decision_timing '{timing}'. Available: {', '.join(DECISION_TIMING_MODES)}")

    encounters = merged.get("encounters", True)
    if isinstance(encounters, str):
        if encounters.strip().lower() not in ("on", "off", "true", "false"):
            raise LiveConfigError("encounters must be on or off")
        encounters = encounters.strip().lower() in ("on", "true")
    if not isinstance(encounters, bool):
        raise LiveConfigError("encounters must be true or false")

    mission_preset = (_clean_str(merged.get("mission_preset"), "mission_preset") or DEFAULT_MISSION_PRESET).lower()
    if mission_preset not in MISSION_PRESETS:
        raise LiveConfigError(f"unknown mission preset '{mission_preset}'. Available: "
                              + ", ".join(f"{k} ({v:g} s)" for k, v in MISSION_PRESETS.items()))
    time_limit = merged.get("time_limit")
    if time_limit is None:
        time_limit = MISSION_PRESETS[mission_preset]
    else:
        try:
            time_limit = float(time_limit)
        except (TypeError, ValueError):
            raise LiveConfigError(f"time_limit must be a number of seconds, got {time_limit!r}") from None
        if not TIME_LIMIT_RANGE[0] <= time_limit <= TIME_LIMIT_RANGE[1]:
            raise LiveConfigError(f"time_limit must be between {TIME_LIMIT_RANGE[0]:g} and {TIME_LIMIT_RANGE[1]:g} seconds")
        mission_preset = next((k for k, v in MISSION_PRESETS.items() if v == time_limit), None)
    mission_clock = (_clean_str(merged.get("mission_clock"), "mission_clock") or DEFAULT_MISSION_CLOCK).lower().replace("-", "_")
    if mission_clock not in MISSION_CLOCK_MODES:
        raise LiveConfigError(f"unknown mission_clock '{mission_clock}'. Available: {', '.join(MISSION_CLOCK_MODES)}")

    mission_timer = (_clean_str(merged.get("mission_timer"), "mission_timer") or DEFAULT_MISSION_TIMER).lower().replace("-", "_")
    commit_policy = (_clean_str(merged.get("commit_policy"), "commit_policy") or "barrier_answer_order").lower().replace("-", "_")
    if commit_policy not in COMMIT_POLICY_CHOICES:
        raise LiveConfigError(f"unknown commit_policy '{commit_policy}'. Available: {', '.join(COMMIT_POLICY_CHOICES)}")
    if mission_timer not in MISSION_TIMERS:
        raise LiveConfigError(f"unknown mission_timer '{mission_timer}'. Available: {', '.join(MISSION_TIMERS)}")
    spawn = (_clean_str(merged.get("spawn"), "spawn") or "different_rooms").lower().replace("-", "_")
    if spawn not in ("different_rooms", "same_room"):
        raise LiveConfigError(f"unknown spawn '{spawn}'. Available: different_rooms, same_room")
    items = (_clean_str(merged.get("items"), "items") or "contested").lower()
    if items not in ("contested", "shared"):
        raise LiveConfigError(f"unknown items '{items}'. Available: contested, shared")
    puzzle = (_clean_str(merged.get("puzzle"), "puzzle") or "riddle").lower()
    if puzzle not in ("none", "split", "riddle"):
        raise LiveConfigError(f"unknown puzzle '{puzzle}'. Available: riddle, split, none")
    policy_dataset = _clean_str(merged.get("policy_dataset"), "policy_dataset") or DEFAULT_POLICY_DATASET
    emulator = (_clean_str(merged.get("policy_emulator"), "policy_emulator") or "table").lower()
    if emulator not in POLICY_EMULATORS and emulator not in POLICY_EMULATORS.values():
        raise LiveConfigError(f"unknown policy_emulator '{emulator}'. Available: {', '.join(POLICY_EMULATORS)}")
    target = (_clean_str(merged.get("policy_target"), "policy_target") or DEFAULT_POLICY_TARGET).lower()
    if target not in POLICY_TARGETS and target not in POLICY_TARGETS.values():
        raise LiveConfigError(f"unknown policy_target '{target}'. Available: {', '.join(POLICY_TARGETS)}")
    surrogate_seed = merged.get("surrogate_seed")
    if surrogate_seed is not None:
        try:
            surrogate_seed = int(surrogate_seed)
        except (TypeError, ValueError):
            raise LiveConfigError(f"surrogate_seed must be an integer, got {surrogate_seed!r}") from None
    hybrid = _hybrid_settings(merged, specs, runtime, timing)
    return LiveConfig(
        **hybrid,
        policy_dataset=policy_dataset, policy_emulator=POLICY_EMULATORS.get(emulator, emulator), surrogate_seed=surrogate_seed,
        policy_target=POLICY_TARGETS.get(target, target),
        black=specs[constants.BLACK_SPY_ID], white=specs[constants.WHITE_SPY_ID], runtime=runtime,
        region=region, aws_profile=aws_profile, timeout_seconds=timeout, openai_temperature=temperature,
        bedrock_temperature=bedrock_temperature,
        seed=seed, decision_timing=timing, personality_preset=preset, encounters=encounters,
        time_limit=time_limit, mission_preset=mission_preset, mission_clock=mission_clock,
        mission_timer=mission_timer, commit_policy=commit_policy, spawn=spawn, items=items, puzzle=puzzle, lives=_on_off(merged.get("lives", True), "lives"), sources=sources,
    )


def _hybrid_spec(actor_id: str, merged: dict[str, Any]) -> ActorSpec:
    """A hybrid spy: the surrogate of a Phase-A identity, escalating to that same identity's real model.
    The provider / model default to the identity's own; anything else is refused (no model substitution)."""
    from workshop.policy_emulator import schema as policy_schema
    prefix, label = ACTOR_PREFIX[actor_id], ACTOR_LABELS[actor_id]
    policy = _clean_str(merged.get(f"{prefix}_policy"), f"{prefix}_policy")
    if policy is None:
        raise LiveConfigError(f"{label}: a hybrid spy needs --{prefix}-policy <model>:<personality>, e.g. gpt61sol:opportunist")
    model_key, sep, personality = policy.partition(":")
    if not sep or model_key not in policy_schema.MODEL_BY_NAME or personality not in VALID_PERSONALITY_PROFILES:
        raise LiveConfigError(f"{label}: '{policy}' is not a policy key <model>:<personality> of a trained identity")
    if merged.get(f"{prefix}_personality") not in (None, "", personality):
        raise LiveConfigError(f"{label}: the policy {policy} plays the {personality} personality, "
                              f"not {merged.get(f'{prefix}_personality')}")
    expected = policy_schema.MODEL_BY_NAME[model_key]["model_id"]
    provider = (_clean_str(merged.get(f"{prefix}_provider") or merged.get("provider"), "provider") or "bedrock").lower()
    model = _clean_str(merged.get(f"{prefix}_model"), f"{prefix}_model") or expected
    if provider != "bedrock" or model != expected:
        raise LiveConfigError(f"{label}: the hybrid escalates to the identity it emulates ({policy} = bedrock / "
                              f"{expected}), not {provider} / {model}")
    return ActorSpec(actor_id=actor_id, provider=provider, model=_validate_model(model, label, provider),
                     personality=personality, controller="hybrid", policy=policy,
                     request_fields=_request_fields(merged.get(f"{prefix}_request_fields") or merged.get("request_fields"),
                                                    label, provider))
def _hybrid_settings(merged: dict[str, Any], specs: dict[str, ActorSpec], runtime: str, timing: str) -> dict[str, Any]:
    hybrid = [s for s in specs.values() if s.controller == "hybrid"]
    router = _clean_str(merged.get("hybrid_router"), "hybrid_router")
    caps = {}
    for key in HYBRID_BUDGET_KEYS:
        value = merged.get(key)
        if value in (None, ""):
            caps[key] = None
            continue
        if isinstance(value, bool):
            raise LiveConfigError(f"{key} must be a non-negative integer")
        try:
            caps[key] = int(value)
        except (TypeError, ValueError):
            raise LiveConfigError(f"{key} must be a non-negative integer, got {value!r}") from None
        if caps[key] < 0:
            raise LiveConfigError(f"{key} must be a non-negative integer")
    if not hybrid:
        if router is not None or any(v is not None for v in caps.values()):
            raise LiveConfigError("--hybrid-router and the LLM-call caps apply to hybrid spies only")
        return {}
    if router is None:
        raise LiveConfigError(f"a hybrid spy needs --hybrid-router ({', '.join(HYBRID_ROUTERS)}); there is no default router")
    router = router.upper()
    if router not in HYBRID_ROUTERS:
        raise LiveConfigError(f"unknown hybrid_router '{router}'. Available: {', '.join(HYBRID_ROUTERS)}")
    if runtime != "native":
        raise LiveConfigError("hybrid spies delegate to the Native LLM controller: use --runtime native")
    if timing != "evaluation":
        raise LiveConfigError("hybrid spies run with --timing evaluation (v1: provider latency spends no mission time)")
    return {"hybrid_router": router, **caps}
def _non_llm_spec(actor_id: str, controller: str, merged: dict[str, Any], personality: str) -> ActorSpec:
    """A surrogate or scripted spy: no provider, no model call. Provider settings given for it are refused."""
    prefix, label = ACTOR_PREFIX[actor_id], ACTOR_LABELS[actor_id]
    for key in (f"{prefix}_provider", f"{prefix}_model", f"{prefix}_request_fields"):
        if merged.get(key) not in (None, ""):
            raise LiveConfigError(f"{label}: --{key.replace('_', '-')} does not apply to a {controller} spy (no provider is called)")
    policy = _clean_str(merged.get(f"{prefix}_policy"), f"{prefix}_policy")
    if controller == "surrogate":
        if policy is None:
            raise LiveConfigError(f"{label}: a surrogate spy needs --{prefix}-policy <model>:<personality>, "
                                  "e.g. gpt61sol:opportunist (see workshop/policies/)")
        model, sep, policy_personality = policy.partition(":")
        if not sep or not model or policy_personality not in VALID_PERSONALITY_PROFILES:
            raise LiveConfigError(f"{label}: '{policy}' is not a policy key <model>:<personality>")
        if merged.get(f"{prefix}_personality") not in (None, "", policy_personality):
            raise LiveConfigError(f"{label}: the policy {policy} plays the {policy_personality} personality, "
                                  f"not {merged.get(f'{prefix}_personality')}")
        return ActorSpec(actor_id=actor_id, provider="surrogate", model=policy, personality=policy_personality,
                         controller=controller, policy=policy)
    if policy is not None:
        raise LiveConfigError(f"{label}: --{prefix}-policy applies to a surrogate spy only")
    return ActorSpec(actor_id=actor_id, provider="scripted", model=SCRIPTED_MODEL, personality=personality,
                     controller=controller)


def _request_fields(value: Any, label: str, provider: str) -> str | None:
    """A JSON object of extra Bedrock request fields, as canonical JSON (or None)."""
    if value in (None, ""):
        return None
    if provider != "bedrock":
        raise LiveConfigError(f"{label}: request fields are a Bedrock setting (additionalModelRequestFields)")
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        raise LiveConfigError(f"{label}: request fields must be a JSON object, e.g. {{\"reasoning\": {{\"effort\": \"low\"}}}}") from None
    if not isinstance(parsed, dict) or not parsed:
        raise LiveConfigError(f"{label}: request fields must be a non-empty JSON object")
    text = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    if len(text) > 500 or _SECRET_KEY_RE.search(text) or _SECRET_VALUE_RE.search(text):
        raise LiveConfigError(f"{label}: request fields must be short model settings, never credentials")
    return text


def _on_off(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("on", "off", "true", "false"):
        return value.strip().lower() in ("on", "true")
    raise LiveConfigError(f"{key} must be on or off")


def _resolve_personalities(merged: dict[str, Any]) -> tuple[str | None, dict[str, str]]:
    """Preset first, then a per-spy personality wins for that spy. The preset
    is reported only if neither spy overrides it."""
    preset = (_clean_str(merged.get("personality_preset"), "personality_preset") or DEFAULT_PERSONALITY_PRESET).lower()
    if preset not in PERSONALITY_PRESETS:
        raise LiveConfigError(f"unknown personality preset '{preset}'. Available: {', '.join(PERSONALITY_PRESETS)}")
    black, white = PERSONALITY_PRESETS[preset]
    chosen = {constants.BLACK_SPY_ID: black, constants.WHITE_SPY_ID: white}
    overridden = False
    for actor_id in ACTORS:
        value = _clean_str(merged.get(f"{ACTOR_PREFIX[actor_id]}_personality"), f"{ACTOR_PREFIX[actor_id]}_personality")
        if value is None:
            continue
        try:
            chosen[actor_id] = validate_personality_profile(value)
        except InvalidPersonalityProfileError:
            raise LiveConfigError(
                f"{ACTOR_LABELS[actor_id]}: unknown personality '{value}'. Available: {', '.join(PERSONALITIES)}"
            ) from None
        overridden = True
    return (None if overridden else preset), chosen
