"""3B / 3C - B0 frequency policy, B1 opportunity-conditioned table policy, target policy v1.

All three are JSON tables of counts fitted on CLEAN full-match decisions of one
policy identity (model + personality). Prediction is deterministic; only the
final sample uses an explicit, seeded random generator.

Action prediction:

    hierarchical table estimate (exact bucket -> backoff -> policy-global)
        -> legal-action masking -> renormalize -> (sample)

The smoothed estimate at a level is (counts + alpha * parent) / (n + alpha), from
the policy-global frequency (itself smoothed towards uniform) down to the
deepest bucket that has at least `min_count` training decisions. The matched
bucket, its backoff level and its sample count are returned with every
prediction, and so is whether masking changed the top action.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from workshop.policy_emulator import schema

# B1 conditioning dimensions, most important first; backoff drops them from the end
B1_DIMENSIONS = ("opponent_alert_present", "inspect_legal", "search_legal", "place_trap_legal",
                 "approach_opponent_legal", "avoid_opponent_legal", "opponent_visible", "mission_complete",
                 "health_band", "life_balance", "time_pressure")
TARGET_DIMENSIONS = ("mission_complete", "opponent_alert_present")
DEFAULT_MIN_COUNT = 5
DEFAULT_ALPHA = 2.0
UNIFORM_ALPHA = 1.0  # pseudo-count towards uniform at the policy-global level


def bucket_key(features: dict[str, Any], dims: tuple[str, ...]) -> str:
    return "|".join(f"{d}={features.get(d)}" for d in dims) or "*"


def legal_action_set(payload: dict[str, Any]) -> list[str]:
    """Actions with at least one legal target (or untargeted), in the policy's action order."""
    legal = schema.legal_targets(payload)
    return [a for a in schema.ACTIONS if a in legal]


def mask_and_renormalize(probs: dict[str, float], legal: list[str]) -> dict[str, float]:
    kept = {a: probs.get(a, 0.0) for a in legal}
    total = sum(kept.values())
    if total <= 0:
        return {a: 1.0 / len(legal) for a in legal} if legal else {}
    return {a: p / total for a, p in kept.items()}


def _top(probs: dict[str, float]) -> str | None:
    # deterministic tie-break: the policy's action order
    return max(probs, key=lambda a: (probs[a], -schema.ACTIONS.index(a) if a in schema.ACTIONS else 0)) if probs else None


@dataclass
class Prediction:
    raw: dict[str, float]  # before masking
    probs: dict[str, float]  # masked, renormalized
    top_action: str | None
    raw_top_action: str | None
    masking_changed_top: bool
    matched_bucket: str
    backoff_level: int
    sample_count: int
    # model-specific explanation (B2: top features; B3: leaf, path, leaf samples); empty for B0/B1
    explanation: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """How concentrated the masked distribution is: its largest probability (not coverage)."""
        return max(self.probs.values()) if self.probs else 0.0

    @property
    def entropy(self) -> float:
        import math
        return -sum(p * math.log2(p) for p in self.probs.values() if p > 0)

    def as_dict(self) -> dict[str, Any]:
        return {"probs": {a: round(p, 4) for a, p in sorted(self.probs.items(), key=lambda x: -x[1])},
                "top_action": self.top_action, "raw_top_action": self.raw_top_action,
                "masking_changed_top": self.masking_changed_top, "matched_bucket": self.matched_bucket,
                "backoff_level": self.backoff_level, "sample_count": self.sample_count,
                "confidence": round(self.confidence, 4), "entropy_bits": round(self.entropy, 4),
                **({"explanation": self.explanation} if self.explanation else {})}


@dataclass
class TablePolicy:
    """B1 (dims = B1_DIMENSIONS) or B0 (dims = (): policy-global frequency only)."""
    dims: tuple[str, ...]
    min_count: int = DEFAULT_MIN_COUNT
    alpha: float = DEFAULT_ALPHA
    actions: tuple[str, ...] = schema.ACTIONS
    # tables[level][bucket] = Counter(action); level = number of dims kept (0 = global)
    tables: dict[int, dict[str, Counter]] = field(default_factory=dict)
    n_train: int = 0

    @property
    def emulator(self) -> str:
        return schema.TABLE_POLICY_VERSION if self.dims else schema.FREQUENCY_POLICY_VERSION

    def fit(self, rows: list[dict]) -> "TablePolicy":
        self.tables = {k: defaultdict(Counter) for k in range(len(self.dims) + 1)}
        for row in rows:
            action = row["intended_action"]
            if action not in self.actions:
                continue
            for k in range(len(self.dims) + 1):
                self.tables[k][bucket_key(row["features"], self.dims[:k])][action] += 1
            self.n_train += 1
        self.tables = {k: dict(v) for k, v in self.tables.items()}
        return self

    def raw_distribution(self, features: dict[str, Any]) -> tuple[dict[str, float], str, int, int]:
        n_actions = len(self.actions)
        probs = {a: 1.0 / n_actions for a in self.actions}
        matched, level_used, count_used = "*", len(self.dims), 0
        for k in range(len(self.dims) + 1):
            key = bucket_key(features, self.dims[:k])
            counts = self.tables.get(k, {}).get(key)
            n = sum(counts.values()) if counts else 0
            if k > 0 and n < self.min_count:
                break
            alpha = UNIFORM_ALPHA if k == 0 else self.alpha
            probs = {a: ((counts or {}).get(a, 0) + alpha * probs[a]) / (n + alpha) for a in self.actions}
            matched, level_used, count_used = key, len(self.dims) - k, n
        return probs, matched, level_used, count_used

    def predict(self, features: dict[str, Any], legal: list[str]) -> Prediction:
        raw, matched, backoff, count = self.raw_distribution(features)
        probs = mask_and_renormalize(raw, legal)
        top, raw_top = _top(probs), _top(raw)
        return Prediction(raw=raw, probs=probs, top_action=top, raw_top_action=raw_top,
                          masking_changed_top=top != raw_top, matched_bucket=matched, backoff_level=backoff,
                          sample_count=count)

    def to_json(self) -> dict[str, Any]:
        return {"emulator": self.emulator, "dims": list(self.dims), "min_count": self.min_count, "alpha": self.alpha,
                "uniform_alpha": UNIFORM_ALPHA, "actions": list(self.actions), "n_train": self.n_train,
                "tables": {str(k): {b: dict(sorted(c.items())) for b, c in sorted(v.items())}
                           for k, v in sorted(self.tables.items())}}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TablePolicy":
        policy = cls(dims=tuple(data["dims"]), min_count=data["min_count"], alpha=data["alpha"],
                     actions=tuple(data["actions"]))
        policy.tables = {int(k): {b: Counter(c) for b, c in v.items()} for k, v in data["tables"].items()}
        policy.n_train = data["n_train"]
        return policy


def frequency_policy(rows: list[dict]) -> TablePolicy:
    return TablePolicy(dims=()).fit(rows)


def table_policy(rows: list[dict], min_count: int = DEFAULT_MIN_COUNT, alpha: float = DEFAULT_ALPHA) -> TablePolicy:
    return TablePolicy(dims=B1_DIMENSIONS, min_count=min_count, alpha=alpha).fit(rows)


# -- target policy -----------------------------------------------------------------------------

@dataclass
class TargetChoice:
    distribution: dict[str | None, float]  # over the action's legal targets
    target_class_probs: dict[str, float]
    source: str  # policy bucket / pooled / uniform / single / untargeted
    fallback: bool

    def top(self) -> str | None:
        if not self.distribution:
            return None
        return max(self.distribution, key=lambda t: self.distribution[t])


@dataclass
class TargetPolicy:
    """P(target class | action, context) per policy, pooled across policies as a fallback;
    uniform within a class. Never a target outside the action's legal targets."""
    dims: tuple[str, ...] = TARGET_DIMENSIONS
    min_count: int = 3
    tables: dict[str, dict[str, Counter]] = field(default_factory=dict)  # level key -> action|bucket -> classes
    pooled: dict[str, Counter] = field(default_factory=dict)  # action -> classes (all policies)
    trap_types: Counter = field(default_factory=Counter)

    def fit(self, rows: list[dict], pooled_rows: list[dict] | None = None) -> "TargetPolicy":
        tables: dict[str, dict[str, Counter]] = {"exact": defaultdict(Counter), "action": defaultdict(Counter)}
        for row in rows:
            action, cls = row["intended_action"], row.get("target_class")
            if action is None or cls is None:
                continue
            tables["exact"][f"{action}|{bucket_key(row['features'], self.dims)}"][cls] += 1
            tables["action"][action][cls] += 1
            if action == "place_trap" and row.get("intended_trap_type"):
                self.trap_types[row["intended_trap_type"]] += 1
        self.tables = {k: dict(v) for k, v in tables.items()}
        pooled: dict[str, Counter] = defaultdict(Counter)
        for row in pooled_rows if pooled_rows is not None else rows:
            if row["intended_action"] and row.get("target_class"):
                pooled[row["intended_action"]][row["target_class"]] += 1
        self.pooled = dict(pooled)
        return self

    def choose(self, action: str, payload: dict[str, Any], features: dict[str, Any]) -> TargetChoice:
        targets = schema.legal_targets(payload).get(action, [])
        if not targets or targets == [None]:
            return TargetChoice({None: 1.0}, {"none": 1.0}, "untargeted", False)
        targets = [t for t in targets if t is not None]
        if len(targets) == 1:
            return TargetChoice({targets[0]: 1.0}, {schema.target_class(action, targets[0], payload): 1.0}, "single", False)
        classes: dict[str, list[str]] = defaultdict(list)
        for t in targets:
            classes[schema.target_class(action, t, payload)].append(t)
        for source, counts in ((f"policy:{action}|{bucket_key(features, self.dims)}",
                                self.tables.get("exact", {}).get(f"{action}|{bucket_key(features, self.dims)}")),
                               (f"policy:{action}", self.tables.get("action", {}).get(action)),
                               (f"pooled:{action}", self.pooled.get(action))):
            if not counts:
                continue
            present = {c: n for c, n in counts.items() if c in classes}
            if sum(present.values()) >= (self.min_count if source.startswith("policy:") and "|" in source else 1):
                total = sum(present.values())
                class_probs = {c: n / total for c, n in present.items()}
                dist = {t: class_probs[c] / len(ts) for c, ts in classes.items() if c in class_probs for t in ts}
                return TargetChoice(dist, class_probs, source, False)
        return TargetChoice({t: 1.0 / len(targets) for t in targets},
                            {c: len(ts) / len(targets) for c, ts in classes.items()}, "uniform", True)

    def trap_type(self, payload: dict[str, Any], rng: random.Random) -> str | None:
        available = sorted(t for t, n in ((payload.get("self") or {}).get("trap_inventory") or {}).items() if n)
        if not available:
            return None
        weights = [self.trap_types.get(t, 0) + 1 for t in available]  # empirical + 1, available types only
        return rng.choices(available, weights=weights, k=1)[0]

    def to_json(self) -> dict[str, Any]:
        return {"emulator": schema.TARGET_POLICY_VERSION, "dims": list(self.dims), "min_count": self.min_count,
                "within_class": "uniform over the legal targets of that class",
                "tables": {lvl: {k: dict(sorted(c.items())) for k, c in sorted(v.items())} for lvl, v in self.tables.items()},
                "pooled": {k: dict(sorted(c.items())) for k, c in sorted(self.pooled.items())},
                "trap_types": dict(sorted(self.trap_types.items()))}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TargetPolicy":
        tp = cls(dims=tuple(data["dims"]), min_count=data["min_count"])
        tp.tables = {lvl: {k: Counter(c) for k, c in v.items()} for lvl, v in data["tables"].items()}
        tp.pooled = {k: Counter(c) for k, c in data["pooled"].items()}
        tp.trap_types = Counter(data.get("trap_types") or {})
        return tp
