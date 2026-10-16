"""Stage 3 milestone 4: the hybrid router (hybrid-router-v1). No provider call.

At each decision point of a hybrid spy the router sees only what the surrogate
sees: the policy features of the live payload, the legal actions, B2's masked
action distribution and its proposed (sampled) action, the frozen coverage
estimator's assessment, B1's masked distribution on the same state (a
diagnostic expert, never sampled, never a fallback) and the actions the spy
committed so far. It returns a RouterResult; escalate=True hands this one
decision to the real LLM controller of the same identity.

Signals (thresholds fixed by the milestone, before any hybrid match):
    repetition_loop            the previous 4 committed decisions used the same action and B2 proposes it again
    strong_b1_b2_disagreement  TVD(P_B1, P_B2) > 0.5, both masked to the same executable legal actions
    coverage_ood               coverage label OOD (coverage-v1)
    low_ood_critical           coverage LOW or OOD and a critical state (time pressure above relaxed, or health band low)
    low_high_confidence        coverage LOW and B2 confidence >= 0.8

Routers:
    R0  never escalates (reference)
    R1  repetition_loop or coverage_ood
    R2  repetition_loop or strong_b1_b2_disagreement or low_ood_critical
    R3  R2 or low_high_confidence (diagnostic)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from workshop.policy_emulator import schema
from workshop.policy_emulator.features import FEATURE_SCHEMA_VERSION

ROUTER_VERSION = "hybrid-router-v1"
BASE_EMULATOR = "logistic-v1"
DIAGNOSTIC_EMULATOR = "table-v1"
TARGET_POLICY = "target-v2.1"
COVERAGE = "coverage-v1"
REPETITION_WINDOW = 4
TVD_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.8
SIGNALS = ("repetition_loop", "strong_b1_b2_disagreement", "coverage_ood", "low_ood_critical", "low_high_confidence")
ROUTERS: dict[str, tuple[str, ...]] = {
    "R0": (),
    "R1": ("repetition_loop", "coverage_ood"),
    "R2": ("repetition_loop", "strong_b1_b2_disagreement", "low_ood_critical"),
    "R3": ("repetition_loop", "strong_b1_b2_disagreement", "low_ood_critical", "low_high_confidence"),
}
ROUTER_PURPOSE = {
    "R0": "reference: always B2",
    "R1": "minimal: very low escalation rate, highly interpretable",
    "R2": "evidence router: the strongest observed failure regions without escalating every LOW state",
    "R3": "diagnostic: R2 plus confident extrapolation in LOW coverage",
}
CRITICAL_DEFINITION = ("time_pressure not in (None, relaxed) or health_band == low "
                       "(milestone 3, closed_loop_stability._critical)")

# Stage 3 milestone 4C: hybrid-router-v2. R2.1 drops LOW/OOD + critical as a trigger (no local benefit as the only
# reason in the 8-match R2 pilot - development evidence, not validation). Signal definitions and thresholds are
# unchanged; the other signals are still computed and logged, as diagnostics only.
ROUTER_VERSION_V2 = "hybrid-router-v2"
ROUTERS_V2: dict[str, tuple[str, ...]] = {
    "R0": (),
    "R2.1": ("repetition_loop", "strong_b1_b2_disagreement"),
}
DIAGNOSTIC_SIGNALS_V2 = ("coverage_ood", "low_ood_critical", "low_high_confidence")
ROUTER_PURPOSE_V2 = {
    "R0": "reference: always B2",
    "R2.1": "R2 without the LOW/OOD + critical trigger: repetition loops and strong B1/B2 disagreement only",
}
DERIVATION_V2 = {"from_router": "R2", "from_version": ROUTER_VERSION,
                 "change": "low_ood_critical removed as a trigger; still computed and logged",
                 "development_evidence": "workshop/policy_reports/phase-a-v2/hybrid_pilot_report.html (8-match R2 "
                                         "pilot; critical-only escalations: 46, LLM changed the action in 24%, no "
                                         "local TVD gain). Development evidence only, not validation."}


def version_of(router_id: str) -> str:
    """The artifact a router id belongs to (R0 is the same in both; it stays with v1)."""
    return ROUTER_VERSION if router_id in ROUTERS or router_id not in ROUTERS_V2 else ROUTER_VERSION_V2


def router_signals(router_id: str) -> tuple[str, ...]:
    if router_id in ROUTERS:
        return ROUTERS[router_id]
    if router_id in ROUTERS_V2:
        return ROUTERS_V2[router_id]
    raise ValueError(f"unknown router {router_id!r}; available: {', '.join(all_router_ids())}")


def all_router_ids() -> tuple[str, ...]:
    return tuple(ROUTERS) + tuple(r for r in ROUTERS_V2 if r not in ROUTERS)


def critical_state(features: dict[str, Any]) -> bool:
    """Frozen in milestone 3 (the closed-loop hybrid-readiness analysis, closed_loop_stability._critical),
    copied verbatim and not redefined here: any time pressure other than relaxed (tight, critical, or the
    extractor's "unknown") or a low health band."""
    return features.get("time_pressure") not in (None, "relaxed") or features.get("health_band") == "low"


def tvd(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _top(p: dict[str, float]) -> str | None:
    if not p:
        return None
    order = list(schema.ACTIONS)
    return max(p, key=lambda a: (p[a], -order.index(a) if a in order else 0))


def repetition_count(history: list[str], proposed: str | None) -> int:
    """How many of the most recent committed decisions used `proposed` in a row."""
    n = 0
    for action in reversed(history):
        if action != proposed:
            break
        n += 1
    return n


@dataclass(frozen=True)
class RouterInputs:
    features: dict[str, Any]
    b2_probs: dict[str, float]          # masked to the executable legal actions
    b2_proposed: str | None             # B2's sampled action for this decision (not committed yet)
    b2_confidence: float                # the largest masked B2 probability
    coverage: dict[str, Any]            # coverage-v1 assessment (label under "coverage")
    b1_probs: dict[str, float] | None   # masked to the same executable legal actions; None = not computed
    history: tuple[str, ...] = ()       # actions committed so far by this spy (both sources), oldest first


@dataclass
class RouterResult:
    escalate: bool
    router: str
    reasons: list[str]
    signals: dict[str, bool]
    b2_confidence: float
    b2_top: str | None
    b2_proposed: str | None
    coverage: str | None
    coverage_score: float | None
    coverage_reasons: list[str]
    b1_b2_tvd: float | None
    b1_top: str | None
    top_actions_differ: bool | None
    critical_state: bool
    repetition_count: int
    repeated_action: str | None
    budget_exhausted: bool = False
    version: str = ROUTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_signals(x: RouterInputs) -> tuple[dict[str, bool], dict[str, Any]]:
    label = (x.coverage or {}).get("coverage")
    critical = critical_state(x.features)
    reps = repetition_count(list(x.history), x.b2_proposed)
    d = round(tvd(x.b1_probs, x.b2_probs), 6) if x.b1_probs is not None else None
    signals = {
        "repetition_loop": x.b2_proposed is not None and reps >= REPETITION_WINDOW,
        "strong_b1_b2_disagreement": d is not None and d > TVD_THRESHOLD,
        "coverage_ood": label == "OOD",
        "low_ood_critical": label in ("LOW", "OOD") and critical,
        "low_high_confidence": label == "LOW" and x.b2_confidence >= CONFIDENCE_THRESHOLD,
    }
    detail = {"coverage": label, "critical": critical, "repetition_count": reps, "tvd": d,
              "b1_top": _top(x.b1_probs) if x.b1_probs is not None else None, "b2_top": _top(x.b2_probs)}
    return signals, detail


def route(router_id: str, x: RouterInputs, *, budget_available: bool = True) -> RouterResult:
    rules = router_signals(router_id)
    signals, d = evaluate_signals(x)
    reasons = [s for s in rules if signals[s]]
    wants = bool(reasons)
    cov = x.coverage or {}
    return RouterResult(
        escalate=wants and budget_available, router=router_id, reasons=reasons, signals=signals,
        b2_confidence=round(x.b2_confidence, 6), b2_top=d["b2_top"], b2_proposed=x.b2_proposed,
        coverage=d["coverage"], coverage_score=cov.get("score"), coverage_reasons=list(cov.get("reasons") or []),
        b1_b2_tvd=d["tvd"], b1_top=d["b1_top"],
        top_actions_differ=None if d["b1_top"] is None else d["b1_top"] != d["b2_top"],
        critical_state=d["critical"], repetition_count=d["repetition_count"],
        repeated_action=x.b2_proposed if signals["repetition_loop"] else None,
        budget_exhausted=wants and not budget_available, version=version_of(router_id))


# -- the router protocol artifact --------------------------------------------------------------------

def _artifact_root(dataset: str, root: Path | None = None, version: str = ROUTER_VERSION) -> Path:
    from workshop.policy_emulator.artifacts import POLICY_ROOT
    from workshop.runners.common import PROJECT_ROOT
    return (root or PROJECT_ROOT / POLICY_ROOT) / dataset / version


def build_protocol(dataset: str = "phase-a-v2", root: Path | None = None, version: str = ROUTER_VERSION) -> dict[str, Any]:
    """Rules, thresholds and the hashes of every artifact the router and its base path read."""
    from workshop.policy_emulator.artifacts import content_hash
    if version == ROUTER_VERSION_V2:
        v1 = build_protocol(dataset, root, ROUTER_VERSION)
        body = {k: v for k, v in v1.items() if k not in ("router_hash", "routers", "purpose", "router_version")}
        body.update({"router_version": ROUTER_VERSION_V2, "routers": {k: list(v) for k, v in ROUTERS_V2.items()},
                     "purpose": ROUTER_PURPOSE_V2, "diagnostic_signals": list(DIAGNOSTIC_SIGNALS_V2),
                     "escalating_signals": sorted({s for v in ROUTERS_V2.values() for s in v}),
                     "derived_from": {**DERIVATION_V2, "from_router_hash": v1["router_hash"]}})
        return {**body, "router_hash": content_hash(body)}
    if version != ROUTER_VERSION:
        raise ValueError(f"unknown router version {version!r}")
    return _build_v1(dataset, root, content_hash)


def _build_v1(dataset: str, root: Path | None, content_hash) -> dict[str, Any]:
    from workshop.policy_emulator.artifacts import load_policy
    identities = {}
    for key in schema.all_policy_keys():
        b2 = load_policy(dataset, key, emulator=BASE_EMULATOR, target=TARGET_POLICY, root=root)
        b1 = load_policy(dataset, key, emulator=DIAGNOSTIC_EMULATOR, target=TARGET_POLICY, coverage=False, root=root)
        identities[key] = {"base_emulator_hash": b2.metadata["content_hash"],
                           "target_policy_hash": (b2.target_metadata or {})["content_hash"],
                           "coverage_hash": b2.coverage_hash, "b1_hash": b1.metadata["content_hash"],
                           "training_dataset_hash": b2.metadata.get("training_dataset_hash")}
    body = {
        "router_version": ROUTER_VERSION, "dataset": dataset,
        "base_emulator": BASE_EMULATOR, "target_policy": TARGET_POLICY, "coverage_version": COVERAGE,
        "diagnostic_emulator": DIAGNOSTIC_EMULATOR, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "signals": list(SIGNALS), "routers": {k: list(v) for k, v in ROUTERS.items()}, "purpose": ROUTER_PURPOSE,
        "thresholds": {"repetition_window": REPETITION_WINDOW, "b1_b2_tvd_above": TVD_THRESHOLD,
                       "low_coverage_confidence_at_least": CONFIDENCE_THRESHOLD},
        "critical_state": CRITICAL_DEFINITION,
        "masking": "B1 and B2 distributions masked to the same executable legal actions (legal minus actions the "
                   "surrogate cannot carry out, e.g. enter_safe_code without a code)",
        "observable_inputs": ["policy features of the live payload (policy-features-v1)", "legal actions",
                              "B2 masked distribution + sampled proposal", "coverage-v1 assessment",
                              "B1 masked distribution", "the spy's own committed actions"],
        "identities": identities,
    }
    return {**body, "router_hash": content_hash(body)}


def save_protocol(dataset: str = "phase-a-v2", root: Path | None = None, version: str = ROUTER_VERSION) -> Path:
    out = _artifact_root(dataset, root, version)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "router.json"
    path.write_text(json.dumps(build_protocol(dataset, root, version), indent=1, sort_keys=True) + "\n")
    return path


def protocol_for(router_id: str, dataset: str = "phase-a-v2", root: Path | None = None) -> dict[str, Any]:
    """The checked protocol artifact a router id belongs to."""
    return load_protocol(dataset, root, version_of(router_id))


def load_protocol(dataset: str = "phase-a-v2", root: Path | None = None, version: str = ROUTER_VERSION) -> dict[str, Any]:
    """The stored protocol, checked against the artifacts on disk (any drift is an error)."""
    from workshop.policy_emulator.artifacts import ArtifactProvenanceError
    path = _artifact_root(dataset, root, version) / "router.json"
    if not path.is_file():
        script = "hybrid_router_replay" if version == ROUTER_VERSION else "hybrid_router_v2"
        raise ArtifactProvenanceError(f"{path} missing: python -m workshop.scripts.{script} --write-protocol")
    stored = json.loads(path.read_text())
    current = build_protocol(dataset, root, version)
    if stored.get("router_hash") != current["router_hash"]:
        raise ArtifactProvenanceError(f"{path}: router hash {stored.get('router_hash')} does not match the current "
                                      f"artifacts ({current['router_hash']})")
    return stored
