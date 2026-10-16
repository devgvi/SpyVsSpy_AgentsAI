"""PolicyCoverageEstimator (coverage-v1): is this state well represented in one
policy's training data? Usable with B0-B3 (it never looks at a model).

Coverage is not confidence:
    confidence = how concentrated the model's action distribution is;
    coverage   = how much training support the state's region has.
High confidence with low coverage is extrapolation; low confidence with high
coverage is a genuinely ambiguous, familiar state.

Signals, all counted on the policy's CLEAN full-match training decisions:
* exact support: training decisions in the state's exact B1 bucket;
* nearest support: training decisions whose B1 bucket differs in at most one
  of the 11 B1 dimensions;
* core patterns: time pressure; health band (overall and while alerted); life
  balance while the opponent is visible; the opponent-state pattern
  (visible, alert, distance bucket, life balance) while alerted; the
  legal-action-set pattern;
* categorical values never seen (they would hit the encoder's __other__);
* numeric features outside the training range.

Labels (rules, in this order):
    OOD     a core pattern with zero training support
    LOW     exact support < 5, a core pattern seen fewer than 5 times, an unseen
            legal-action-set pattern, or two or more numeric features out of range
    MEDIUM  exact support < 20, nearest support < 50, one numeric feature out of
            range, or an unseen categorical value
    HIGH    otherwise

`score` is a heuristic in [0, 1] (log support, halved per LOW reason, x0.75 per
MEDIUM reason, at most 0.05 when OOD). It is not a calibrated probability that
the surrogate is right.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from workshop.policy_emulator.linear import CATEGORICAL, NUMERIC
from workshop.policy_emulator.policies import B1_DIMENSIONS, bucket_key, legal_action_set

COVERAGE_VERSION = "coverage-v1"
RARE = 5
EXACT_LOW, EXACT_MEDIUM, NEAR_MEDIUM = 5, 20, 50
LABELS = ("HIGH", "MEDIUM", "LOW", "OOD")


def _opponent_pattern(f: dict[str, Any]) -> str:
    return f"visible={f['opponent_visible']}|alert={f['opponent_alert_present']}|" \
           f"distance={f['opponent_distance_bucket']}|life={f['life_balance']}"


@dataclass
class PolicyCoverageEstimator:
    n: int = 0
    buckets: Counter = field(default_factory=Counter)
    patterns: dict[str, Counter] = field(default_factory=dict)
    values: dict[str, Counter] = field(default_factory=dict)
    ranges: dict[str, list[float]] = field(default_factory=dict)

    version = COVERAGE_VERSION

    def fit(self, rows: list[dict]) -> "PolicyCoverageEstimator":
        self.n = len(rows)
        self.buckets = Counter(bucket_key(r["features"], B1_DIMENSIONS) for r in rows)
        p = {k: Counter() for k in ("time_pressure", "health_band", "health_band_alerted", "life_balance_visible",
                                    "opponent_state_alerted", "legal_set")}
        for r in rows:
            f = r["features"]
            p["time_pressure"][f["time_pressure"]] += 1
            p["health_band"][f["health_band"]] += 1
            if f["opponent_alert_present"]:
                p["health_band_alerted"][f["health_band"]] += 1
                p["opponent_state_alerted"][_opponent_pattern(f)] += 1
            if f["opponent_visible"]:
                p["life_balance_visible"][f["life_balance"]] += 1
            p["legal_set"]["+".join(legal_action_set(r["observation"]))] += 1
        self.patterns = p
        self.values = {n: Counter(str(r["features"].get(n)) for r in rows) for n in CATEGORICAL}
        self.ranges = {}
        for n in NUMERIC:
            vals = [r["features"][n] for r in rows if isinstance(r["features"].get(n), int | float)
                    and not isinstance(r["features"].get(n), bool)]
            if vals:
                self.ranges[n] = [min(vals), max(vals)]
        return self

    def _near_support(self, key: str) -> int:
        mine = key.split("|")
        total = 0
        for other, count in self.buckets.items():
            if sum(a != b for a, b in zip(mine, other.split("|"))) <= 1:
                total += count
        return total

    def assess(self, features: dict[str, Any], legal: list[str]) -> dict[str, Any]:
        f = features
        key = bucket_key(f, B1_DIMENSIONS)
        exact, near = self.buckets.get(key, 0), self._near_support(key)
        ood, low, medium = [], [], []
        core = [("time_pressure", f["time_pressure"], "time_pressure"),
                ("health_band", f["health_band"], "health_band")]
        if f["opponent_alert_present"]:
            core.append(("health_band_alerted", f["health_band"], "health_band while alerted"))
            core.append(("opponent_state_alerted", _opponent_pattern(f), "opponent state while alerted"))
        if f["opponent_visible"]:
            core.append(("life_balance_visible", f["life_balance"], "life_balance while visible"))
        unseen = []
        for name, value, label in core:
            count = self.patterns.get(name, {}).get(value, 0)
            if count == 0:
                ood.append(f"{label} = {value}: never seen in training")
                unseen.append(f"{label}={value}")
            elif count < RARE:
                low.append(f"{label} = {value}: {count} training decisions")
        legal_key = "+".join(legal)
        if self.patterns.get("legal_set", {}).get(legal_key, 0) == 0:
            low.append(f"legal-action set {legal_key}: never seen in training")
        if exact < EXACT_LOW:
            low.append(f"exact B1 bucket: {exact} training decisions")
        elif exact < EXACT_MEDIUM:
            medium.append(f"exact B1 bucket: {exact} training decisions")
        if near < NEAR_MEDIUM:
            medium.append(f"nearest support (one B1 dimension away): {near}")
        out_of_range = [f"{n}={f.get(n)} outside [{lo:g}, {hi:g}]" for n, (lo, hi) in self.ranges.items()
                        if isinstance(f.get(n), int | float) and not isinstance(f.get(n), bool) and not lo <= f[n] <= hi]
        if len(out_of_range) >= 2:
            low.append(f"{len(out_of_range)} numeric features outside the training range")
        elif out_of_range:
            medium.append(out_of_range[0])
        new_values = [f"{n}={f.get(n)}" for n in CATEGORICAL if self.values.get(n, {}).get(str(f.get(n)), 0) == 0]
        if new_values:
            medium.append("unseen value: " + ", ".join(new_values))
            unseen += new_values
        label = "OOD" if ood else "LOW" if low else "MEDIUM" if medium else "HIGH"
        score = min(1.0, math.log1p(exact) / math.log1p(50)) * (0.5 ** len(low)) * (0.75 ** len(medium))
        if ood:
            score = min(score, 0.05)
        return {"coverage": label, "score": round(score, 3), "reasons": ood + low + medium,
                "exact_support": exact, "nearest_support": near, "unseen_features": unseen,
                "out_of_range": out_of_range}

    def to_json(self) -> dict[str, Any]:
        return {"emulator": self.version, "n": self.n, "buckets": dict(sorted(self.buckets.items())),
                "patterns": {k: dict(sorted(v.items())) for k, v in self.patterns.items()},
                "values": {k: dict(sorted(v.items())) for k, v in self.values.items()}, "ranges": self.ranges,
                "thresholds": {"rare": RARE, "exact_low": EXACT_LOW, "exact_medium": EXACT_MEDIUM,
                               "near_medium": NEAR_MEDIUM}}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PolicyCoverageEstimator":
        return cls(n=data["n"], buckets=Counter(data["buckets"]),
                   patterns={k: Counter(v) for k, v in data["patterns"].items()},
                   values={k: Counter(v) for k, v in data["values"].items()},
                   ranges={k: list(v) for k, v in data["ranges"].items()})
