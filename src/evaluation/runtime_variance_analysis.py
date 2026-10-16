"""M6A.1: pure, network-free variance-decomposition functions (spec
sections 8-15, 25, 26) - canonical decision signatures, distribution
counting, modal-rate/overlap computation, hash-based snapshot integrity,
and the per-snapshot runtime-effect classification. Nothing here makes a
network call; the live drivers (`run_m6a1.py`) are the only network-facing
code for this milestone.

Core principle (spec section 1): `Native decision != Strands decision` is
not automatically a runtime effect - it is only evidence of one once it
exceeds ordinary within-runtime resampling variance. Every function below
exists to make that comparison explicit rather than asserted.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

# -- canonical decision signature (spec section 8) ---------------------------


def canonical_decision_signature(action: str | None, target: str | None, trap_type: str | None) -> str:
    """`MOVE|door_security`, `PLACE_TRAP|safe_office|electric`,
    `INSPECT|desk_03`, `WAIT` - ignores non-behavioral wording (`reason`
    text) entirely, matching spec section 8's examples. `action=None`
    (a runtime-error trial with no executed action) is its own explicit
    signature, never silently coerced to WAIT."""
    if action is None:
        return "NO_ACTION"
    parts = [action.upper()]
    if target is not None:
        parts.append(str(target))
    if trap_type is not None:
        parts.append(str(trap_type))
    return "|".join(parts)


def trial_signature(trial: dict[str, Any]) -> str:
    return canonical_decision_signature(trial.get("action"), trial.get("target"), trial.get("trap_type"))


# -- distribution / modal / overlap (spec sections 9-13) ---------------------


def build_distribution(trials: list[dict[str, Any]]) -> Counter:
    return Counter(trial_signature(t) for t in trials)


def modal_decision(distribution: Counter) -> tuple[str | None, float]:
    """Returns (signature, rate) for the single most common signature -
    `None, 0.0` for an empty distribution. Ties broken by insertion order
    (Counter.most_common's own documented tie-break), never re-sorted by
    any behavioral preference."""
    total = sum(distribution.values())
    if total == 0:
        return None, 0.0
    signature, count = distribution.most_common(1)[0]
    return signature, count / total


def unique_decision_count(distribution: Counter) -> int:
    return len(distribution)


def distribution_overlap(dist_a: Counter, dist_b: Counter) -> float:
    """Spec section 13: `sum_D min(P_a(D), P_b(D))` over normalized
    probability distributions. 1.0 = identical empirical distributions,
    0.0 = disjoint support. Returns 0.0 (not an error) if either
    distribution is empty - there is no overlap to report."""
    total_a, total_b = sum(dist_a.values()), sum(dist_b.values())
    if total_a == 0 or total_b == 0:
        return 0.0
    keys = set(dist_a) | set(dist_b)
    return sum(min(dist_a.get(k, 0) / total_a, dist_b.get(k, 0) / total_b) for k in keys)


def action_only_distribution(trials: list[dict[str, Any]]) -> Counter:
    """Spec section 15: action_type variation, ignoring target/parameter -
    a coarser distribution than `build_distribution` for separating
    "different action" from "same action, different target"."""
    return Counter((t.get("action") or "NO_ACTION") for t in trials)


# -- per-snapshot runtime-effect classification (spec section 14) -----------

# Documented, fixed thresholds (not tuned post-hoc on this milestone's own
# data): a runtime counts as "internally stable" on a snapshot at
# modal_rate >= STABILITY_THRESHOLD; two distributions count as
# "substantially overlapping" at overlap >= OVERLAP_THRESHOLD. Both are
# round, defensible values chosen before looking at live results, not
# fitted to produce a particular verdict.
STABILITY_THRESHOLD = 0.8
OVERLAP_THRESHOLD_HIGH = 0.8
OVERLAP_THRESHOLD_SUBSTANTIAL = 0.5

NO_DETECTABLE_RUNTIME_EFFECT = "NO_DETECTABLE_RUNTIME_EFFECT"
WITHIN_RUNTIME_VARIANCE_DOMINATES = "WITHIN_RUNTIME_VARIANCE_DOMINATES"
BETWEEN_RUNTIME_SHIFT = "BETWEEN_RUNTIME_SHIFT"
RUNTIME_SPECIFIC_MODE = "RUNTIME_SPECIFIC_MODE"
MIXED = "MIXED"


def classify_snapshot_runtime_effect(native_dist: Counter, strands_dist: Counter) -> str:
    """Spec section 14, priority-ordered (never a blended score - spec
    section 12 forbids collapsing everything into one number):

    1. Same modal decision AND high overlap -> NO_DETECTABLE_RUNTIME_EFFECT
       ("same mode, high overlap").
    2. Both runtimes individually stable (modal_rate >= 0.8) but disagree
       on the mode -> RUNTIME_SPECIFIC_MODE ("each runtime internally
       stable but different modal decision").
    3. Substantial overlap remains (>= 0.5) -> WITHIN_RUNTIME_VARIANCE_DOMINATES
       ("both unstable, strong overlap" - or one unstable enough that the
       apparent difference is plausibly noise).
    4. Low overlap (< 0.5) -> BETWEEN_RUNTIME_SHIFT ("distributions differ
       beyond self-variance").
    5. Empty input -> MIXED (no data to classify; the only reachable case
       not covered by 1-4, kept as an explicit fallback rather than a
       silent default)."""
    total_native, total_strands = sum(native_dist.values()), sum(strands_dist.values())
    if total_native == 0 or total_strands == 0:
        return MIXED

    native_mode, native_rate = modal_decision(native_dist)
    strands_mode, strands_rate = modal_decision(strands_dist)
    overlap = distribution_overlap(native_dist, strands_dist)
    same_modal = native_mode == strands_mode

    if same_modal and overlap >= OVERLAP_THRESHOLD_HIGH:
        return NO_DETECTABLE_RUNTIME_EFFECT
    if native_rate >= STABILITY_THRESHOLD and strands_rate >= STABILITY_THRESHOLD and not same_modal:
        return RUNTIME_SPECIFIC_MODE
    if overlap >= OVERLAP_THRESHOLD_SUBSTANTIAL:
        return WITHIN_RUNTIME_VARIANCE_DOMINATES
    return BETWEEN_RUNTIME_SHIFT


# -- overall interpretation (spec section 48) --------------------------------

RUNTIME_EFFECT_SUPPORTED = "RUNTIME_EFFECT_SUPPORTED"
INTRINSIC_VARIANCE_DOMINATES = "INTRINSIC_VARIANCE_DOMINATES"
RUNTIME_EFFECT_CONTEXT_DEPENDENT = "RUNTIME_EFFECT_CONTEXT_DEPENDENT"
TECHNICAL_COST_DIFFERENCE_ONLY = "TECHNICAL_COST_DIFFERENCE_ONLY"
OVERALL_MIXED = "MIXED"


def classify_overall(per_snapshot_classifications: list[str], cost_differs: bool) -> str:
    """Aggregates the per-snapshot classifications (spec section 48) - a
    second, coarser priority-ordered decision, never a numeric average of
    the per-snapshot labels:

    - Any RUNTIME_SPECIFIC_MODE or BETWEEN_RUNTIME_SHIFT snapshot is real,
      snapshot-level evidence of a runtime effect (spec section 49's
      "strong runtime effect" example is exactly a RUNTIME_SPECIFIC_MODE
      case). If it's most snapshots -> RUNTIME_EFFECT_SUPPORTED; if it's
      some but not most -> RUNTIME_EFFECT_CONTEXT_DEPENDENT (real in some
      decision contexts, not others - spec section 33's own question).
    - If none of those occur anywhere, but cost genuinely differs
      (latency/tokens) -> TECHNICAL_COST_DIFFERENCE_ONLY.
    - If none occur and cost does not meaningfully differ ->
      INTRINSIC_VARIANCE_DOMINATES.
    - Empty input -> MIXED."""
    if not per_snapshot_classifications:
        return OVERALL_MIXED
    evidence = sum(1 for c in per_snapshot_classifications if c in (RUNTIME_SPECIFIC_MODE, BETWEEN_RUNTIME_SHIFT))
    n = len(per_snapshot_classifications)
    if evidence == 0:
        return TECHNICAL_COST_DIFFERENCE_ONLY if cost_differs else INTRINSIC_VARIANCE_DOMINATES
    if evidence >= (n / 2):
        return RUNTIME_EFFECT_SUPPORTED
    return RUNTIME_EFFECT_CONTEXT_DEPENDENT


# -- hash-based integrity (spec section 25) ----------------------------------


def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def stable_hash(obj: Any) -> str:
    """Public alias - M6A.2 (`episode_trajectory_analysis.py`) reuses this
    exact hashing convention for its own state/memory-summary hashes
    rather than inventing a second one."""
    return _stable_hash(obj)


def observation_hash(observation_payload: dict[str, Any]) -> str:
    """Matches `LLMController._request_action`'s own existing digest
    convention exactly (same field excluded, same hash construction) so
    this milestone's hashes are directly comparable to production
    behavior, not a parallel invented scheme."""
    payload = dict(observation_payload)
    payload.pop("time", None)
    return _stable_hash(payload)


def legal_actions_hash(legal_actions: list[dict[str, Any]]) -> str:
    return _stable_hash(legal_actions)


def memory_hash(memory_compact_dict: dict[str, Any]) -> str:
    return _stable_hash(memory_compact_dict)


def content_hash(*parts: str) -> str:
    """Hashes actual text CONTENT (spec section 23: "prompt hash... Do
    not log credentials" - exact-payload-equality verification, not
    structure). Distinct from `payload_shape_hash` below on purpose: a
    real bug found while first using this milestone's own hashing -
    `system_prompt`/`user_prompt` are already-serialized strings, and
    running a bare string through `_shape()` always collapses to just
    `"str"` regardless of content, making a shape-hash of two prompt
    strings identical for every snapshot no matter how different their
    actual observations are. Use this function for prompt/request-text
    equality; reserve `payload_shape_hash` for a real nested dict (like
    `output_schema`) where "structure, not content" is the intended
    comparison."""
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def _shape(obj: Any) -> Any:
    """Recursively reduces a payload to its key/type structure only -
    never its content - for `payload_shape_hash` (spec section 24: audit
    request *structure*, e.g. role sequence/tool definitions/schema
    shape, never prompt content or any chain-of-thought)."""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_shape(v) for v in obj[:1]] if obj else []
    return type(obj).__name__


def payload_shape_hash(payload: dict[str, Any]) -> str:
    return _stable_hash(_shape(payload))


# -- snapshot integrity (spec section 5) -------------------------------------


def assert_snapshot_integrity(snapshot) -> list[str]:
    """Re-derives, from the frozen `Observation`/`legal_actions` alone
    (no live `GameState` needed - the snapshot set is already built and
    frozen by this milestone), the specific facts spec section 5 requires:
    every INSPECT/INTERACT/PLACE_TRAP/DISARM legal-action target matches
    the observation's own `nearby_object_id` (the actual M6A bug this
    milestone exists to guard against - see M6A's own `state.update(0.0)`
    fix), WAIT is always legal, and MOVE targets are exactly the
    interactable/door objects the Observation itself lists. Returns a
    list of violation strings - empty means the snapshot is intact."""
    violations: list[str] = []
    obs = snapshot.observation
    legal = snapshot.legal_actions

    if not any(a.get("action") == "wait" for a in legal):
        violations.append("WAIT is not legal - every snapshot must offer it")

    gated_actions = {"inspect", "interact", "place_trap", "disarm"}
    for entry in legal:
        if entry.get("action") in gated_actions:
            target = entry.get("target_id")
            if target != obs.nearby_object_id:
                violations.append(
                    f"{entry['action']} targets {target!r} but nearby_object_id is {obs.nearby_object_id!r} "
                    "(current_interaction_target likely never recomputed - the M6A state.update(0.0) bug)"
                )

    move_targets = {e["target_id"] for e in legal if e.get("action") == "move"}
    expected_move_targets = {o.id for o in obs.objects if o.interactable or o.type == "door"}
    if "exit_door" in {o.id for o in obs.objects}:
        expected_move_targets.add("exit_door")
    if move_targets != expected_move_targets:
        violations.append(f"MOVE targets {move_targets} != expected interactable/door set {expected_move_targets}")

    return violations
