"""The scientific policy registry (FINAL-2 spec 3-5, 30, 35).

A policy is how an agent decides. Each descriptor names the frozen research implementation it wraps and its final
scientific status from the FINAL-1 release. The frozen artifacts are loaded read-only through the existing,
provenance-checked loaders (artifacts.load_policy, stage5b_protocol.arm_base / bound_protocol, router.protocol_for);
nothing here edits, retrains or re-tunes a policy, and one version is never aliased to another.

Compatibility labels (spec 35): runtime compatibility is not scientific validation.
    EVALUATED                - this policy / provider / model / level / execution combination was evaluated
    COMPATIBLE / NOT EVALUATED - it runs through the same controller contract, but was never evaluated
    EXPERIMENTAL             - it depends on an adapter that has never run against a live provider
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DATASET = "phase-a-v2"
TARGET = "target-v2.1"               # the target policy of every frozen closed-loop surrogate
ROUTER = "R2.1"
IDENTITIES = {"sonnet55": "us.anthropic.claude-sonnet-5-5", "gpt61sol": "us.openai.gpt-6.1-sol"}
IDENTITY_BY_MODEL = {v: k for k, v in IDENTITIES.items()}
DEFAULT_IDENTITY = "sonnet55"
PERSONALITIES = ("cautious", "opportunist", "racer", "saboteur")
EVALUATED = "EVALUATED"
COMPATIBLE = "COMPATIBLE / NOT EVALUATED"
EXPERIMENTAL = "EXPERIMENTAL"
FAKE = "FAKE PROVIDER (offline demo, no model)"
BASELINE = "ENGINE BASELINE (not a research policy)"
B2V2_MANIFEST_HASH = "sha256:42e007a6a260fe5d"
R21_HASH = "sha256:216f3c9a3c4b3f78"

# FINAL-1 statuses (workshop/policies/final-release/claims.json, final_release.HYBRID_V*_STATUS; tested equal)
HYBRID_V1_STATUS = "CONFIRMED / GENERALIZED RESEARCH BASELINE"
HYBRID_V2_STATUS = "MORE EFFICIENT BUT NOT SAFELY PROMOTABLE UNDER THE CONFIRMATORY RULE"
STAGE5C_REDUCTION = "14.6%"
# the UI labels of the two hybrids, exactly as FINAL-2 spec 4 / 30 words them
HYBRID_V1_LABEL = "CONFIRMED / GENERALIZED RESEARCH BASELINE"
HYBRID_V2_LABEL = "MORE EFFICIENT \u2014 NOT SAFELY PROMOTABLE UNDER CONFIRMATORY RULE"


@dataclass(frozen=True)
class PolicyDescriptor:
    id: str
    label: str
    kind: str                     # scripted | surrogate | hybrid | llm
    requires_provider: bool
    scientific_status: str
    summary: tuple[str, ...]      # the concise UI status (spec 30)
    base: str | None = None       # surrogate base: b1 | b2_v1 | b2_v2
    router: str | None = None
    frozen_artifact: str = ""
    evaluated_on: str = ""
    evaluated_levels: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)
    ui_status: str = ""           # the short scientific label shown in the UI (spec 30); default: scientific_status

    @property
    def scientific_label(self) -> str:
        return self.ui_status or self.scientific_status


POLICIES: dict[str, PolicyDescriptor] = {
    "scripted": PolicyDescriptor(
        "scripted", "Scripted bot", "scripted", False, BASELINE,
        ("The engine's scripted opponent", "Not a research policy"),
        frozen_artifact="src/bot_controller.py ScriptedBotController (profile from the personality)"),
    "b1": PolicyDescriptor(
        "b1", "B1 (table-v1)", "surrogate", False, "SUPERSEDED (B2 was selected in Stage 3 M3)",
        ("Frequency-table surrogate", "Superseded by B2 in Stage 3"), base="b1",
        frozen_artifact="workshop/policies/phase-a-v2/<model>/<personality>/table-v1 + target-v2.1 + coverage-v1",
        evaluated_on="Stage 3 M3 closed loop, Level 1", evaluated_levels=("level_01",)),
    "b2_v1": PolicyDescriptor(
        "b2_v1", "B2-v1 (logistic-v1)", "surrogate", False,
        "CONFIRMED SURROGATE COMPONENT OF HYBRID-V1 (not adequate on its own in closed loop)",
        ("Logistic surrogate (Phase-A data)", "Loops / times out on its own under shift"), base="b2_v1",
        frozen_artifact="workshop/policies/phase-a-v2/<model>/<personality>/logistic-v1 + target-v2.1 + coverage-v1",
        evaluated_on="Stage 3 M3 / M4D (Level 1); Stage 4C and 5B (Levels 2-3, through Hybrid R0)",
        evaluated_levels=("level_01", "level_02", "level_03")),
    "hybrid_v1": PolicyDescriptor(
        "hybrid_v1", "Hybrid-v1 / R2.1", "hybrid", True, HYBRID_V1_STATUS,
        ("Confirmed generalized research baseline",), base="b2_v1", router=ROUTER,
        frozen_artifact="B2-v1 + B1 (diagnostic) + router R2.1 (hybrid-router-v2/router.json) + the LLM delegate",
        evaluated_on="Stage 3 M4D (Level 1), Stage 4C and 5C (Levels 2-3): Bedrock, the identity's own model, "
                     "evaluation timing", evaluated_levels=("level_01", "level_02", "level_03"), ui_status=HYBRID_V1_LABEL),
    "b2_v2": PolicyDescriptor(
        "b2_v2", "B2-v2 (candidate C1)", "surrogate", False,
        "SELECTED CANDIDATE COMPONENT (Stage 5A decision A, Stage 5B decision A)",
        ("Logistic surrogate + promoted hard examples", "Used inside Hybrid-v2"), base="b2_v2",
        frozen_artifact=f"workshop/policies/stage5/b2-v2-candidate (manifest {B2V2_MANIFEST_HASH}) + target-v2.1 + coverage-v1",
        evaluated_on="Stage 5B call-free closed loop (Levels 1-3)", evaluated_levels=("level_01", "level_02", "level_03")),
    "hybrid_v2": PolicyDescriptor(
        "hybrid_v2", "Hybrid-v2 / R2.1", "hybrid", True, HYBRID_V2_STATUS,
        (f"{STAGE5C_REDUCTION} fewer provider calls in Stage 5C", "Not safely promotable under frozen confirmatory rule"),
        base="b2_v2", router=ROUTER,
        frozen_artifact="B2-v2 + B1 (diagnostic) + router R2.1 (B2-v2 bound) + the LLM delegate",
        evaluated_on="Stage 5C after Amendment 02 (Levels 2-3): Bedrock, the identity's own model, evaluation timing",
        evaluated_levels=("level_02", "level_03"), ui_status=HYBRID_V2_LABEL),
    "real_llm": PolicyDescriptor(
        "real_llm", "REAL LLM", "llm", True, "BEHAVIOURAL REFERENCE (the REAL arm of every trial)",
        ("Every decision is a model call",), frozen_artifact="src/llm_controller.py LLMController (native runtime)",
        evaluated_on="REAL arm of Stage 3-5 on Bedrock (Claude Sonnet 5.5, GPT-6.1 Sol), Levels 1-3",
        evaluated_levels=("level_01", "level_02", "level_03")),
}
POLICY_IDS = tuple(POLICIES)
# research protocol settings an EVALUATED label requires (Stage 3-5 runs: workshop_runs/*/run.json)
RESEARCH_TIME_LIMIT = 300.0


def identity_for(policy: str, provider: str | None, model: str | None, identity: str | None) -> str:
    """The surrogate identity (model family of the frozen artifacts) a surrogate / hybrid spy plays."""
    if identity:
        return identity
    if provider == "bedrock" and model in IDENTITY_BY_MODEL:
        return IDENTITY_BY_MODEL[model]
    return DEFAULT_IDENTITY


def policy_key(identity: str, personality: str) -> str:
    return f"{identity}:{personality}"


@dataclass(frozen=True)
class Compatibility:
    label: str
    reasons: tuple[str, ...]

    @property
    def evaluated(self) -> bool:
        return self.label == EVALUATED


def compatibility(policy: str, *, provider: str | None, model: str | None, identity: str, level: str, mode: str,
                  time_limit: float, enhanced: bool = False) -> Compatibility:
    d = POLICIES[policy]
    reasons: list[str] = []
    if enhanced:  # (the scripted / fake / anthropic labels below take precedence)
        reasons.append("enhanced gameplay: door traps, trap variants and room dressing are playable-runtime rules "
                       "(never part of Stages 3-5)")
    if d.kind == "scripted":
        return Compatibility(BASELINE, ("the engine's scripted bot was never one of the research policies",))
    if provider == "fake":
        return Compatibility(FAKE, ("the fake provider runs no model: telemetry and plumbing only",))
    if provider == "anthropic":
        return Compatibility(EXPERIMENTAL, ("the Anthropic direct adapter was never run against the live API",))
    if mode != "evaluation":
        reasons.append("playable mode: execution semantics differ from the frozen evaluation protocol")
    if abs(time_limit - RESEARCH_TIME_LIMIT) > 1e-9:
        reasons.append(f"mission time {time_limit:g} s (the research protocol used {RESEARCH_TIME_LIMIT:g} s)")
    if level not in d.evaluated_levels:
        reasons.append(f"{d.label} was not evaluated on {level}")
    if d.requires_provider:
        if provider != "bedrock":
            reasons.append(f"the research ran on Bedrock, not {provider}")
        elif IDENTITY_BY_MODEL.get(model or "") is None:
            reasons.append(f"{model} is not one of the research models ({', '.join(IDENTITIES.values())})")
        if d.kind == "hybrid" and IDENTITIES.get(identity) != model:
            reasons.append(f"the surrogate identity {identity} is not the escalation model's own identity")
    if reasons:
        return Compatibility(COMPATIBLE, tuple(reasons))
    return Compatibility(EVALUATED, (d.evaluated_on,))


@dataclass
class FrozenBundle:
    """The read-only frozen artifacts one spy plays, and their identities (hashes) for the runtime manifest."""
    policy: str
    key: str
    base: Any = None   # LoadedPolicy of the surrogate (B1 / B2-v1 / B2-v2)
    b1: Any = None     # hybrids: the diagnostic B1
    protocol: dict[str, Any] | None = None
    identity: dict[str, Any] = field(default_factory=dict)


def _load(key: str, emulator: str):
    from workshop.policy_emulator import dataset as policy_dataset
    from workshop.policy_emulator.artifacts import load_policy
    expected = policy_dataset.load_provenance(DATASET)["dataset_hash"]
    return load_policy(DATASET, key, emulator=emulator, target=TARGET, expected_dataset_hash=expected)


def _ident(lp) -> dict[str, Any]:
    return {"policy_key": lp.policy_key, "emulator": lp.emulator, "emulator_hash": lp.metadata.get("content_hash"),
            "training_dataset_hash": lp.metadata.get("training_dataset_hash"),
            "target": lp.target_version, "target_hash": (lp.target_metadata or {}).get("content_hash"),
            "coverage_hash": lp.coverage_hash}


def load_frozen(policy: str, key: str) -> FrozenBundle:
    """Loads (never edits) the frozen artifacts of one policy for one identity, provenance-checked on load."""
    d = POLICIES[policy]
    bundle = FrozenBundle(policy, key)
    if d.kind in ("scripted", "llm"):
        return bundle
    from workshop.policy_emulator import router as rt
    from workshop.policy_emulator import stage5b_protocol as s5b
    if d.base == "b1":
        bundle.base = _load(key, rt.DIAGNOSTIC_EMULATOR)
    else:
        b2_v1 = _load(key, rt.BASE_EMULATOR)
        bundle.base = s5b.arm_base("b2v2", b2_v1) if d.base == "b2_v2" else b2_v1
    bundle.identity = {"surrogate": _ident(bundle.base)}
    if d.base == "b2_v2":
        bundle.identity["surrogate"]["b2_v2_manifest_hash"] = B2V2_MANIFEST_HASH
    if d.kind == "hybrid":
        bundle.b1 = _load(key, rt.DIAGNOSTIC_EMULATOR)
        protocol = rt.protocol_for(ROUTER, DATASET)
        if d.base == "b2_v2":
            protocol = s5b.bound_protocol("b2v2", key, bundle.base, protocol)
        bundle.protocol = protocol
        bundle.identity.update({"b1": _ident(bundle.b1), "router": ROUTER, "router_hash": protocol["router_hash"],
                                "router_version": protocol.get("router_version")})
    return bundle


__all__ = ["POLICIES", "POLICY_IDS", "PolicyDescriptor", "compatibility", "Compatibility", "identity_for", "policy_key",
           "load_frozen", "FrozenBundle", "IDENTITIES", "PERSONALITIES", "EVALUATED", "COMPATIBLE", "EXPERIMENTAL", "FAKE",
           "BASELINE", "HYBRID_V1_STATUS", "HYBRID_V2_STATUS", "HYBRID_V1_LABEL", "HYBRID_V2_LABEL"]
