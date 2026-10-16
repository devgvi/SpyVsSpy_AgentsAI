"""Stage 3 milestone 2 - B2 multinomial logistic policy and B3 shallow decision tree.

Both read PolicyFeatureExtractorV1 features only (observable), produce a
distribution over the engine action vocabulary, then go through the same
legal-action masking and renormalization as B0/B1. Both serialize to plain
JSON (coefficients / nodes); nothing is pickled and nothing is refitted at
runtime.

B2 - `LogisticPolicy`: L2-regularized multinomial logistic regression on the
deterministic `linear.Encoder` columns; classes are the actions this policy
used in training, and every engine action keeps an epsilon floor
((1 - eps) * softmax + eps / |actions|) so an action it never chose stays
possible when it is the only legal one.

B3 - `TreePolicy`: CART with Gini impurity, `max_depth` and `min_samples_leaf`.
Splits: boolean (true), categorical (== value), numeric (<= threshold; a null
numeric sorts below every value). Leaf distributions are smoothed towards
the parent node (pseudo-count 2, the root towards uniform) like B1.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from workshop.policy_emulator import schema
from workshop.policy_emulator.linear import BOOLEAN, CATEGORICAL, NUMERIC, Encoder, fit_softmax, predict_softmax
from workshop.policy_emulator.policies import Prediction, _top, mask_and_renormalize

LOGISTIC_VERSION = "logistic-v1"
TREE_VERSION = "tree-v1"
EPSILON = 1e-3
# the agent's own recent decisions (strategy memory); dropped by the no-memory ablation
MEMORY_FEATURES = ("recent_action_1", "recent_action_2", "recent_trap_attempt")


def _subset(features: dict[str, Any], dropped: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in features.items() if k not in dropped} if dropped else features


@dataclass
class LogisticPolicy:
    c: float = 1.0
    class_weight: str = "none"  # none | balanced
    feature_set: str = "full"  # full | no_memory
    epsilon: float = EPSILON
    actions: tuple[str, ...] = schema.ACTIONS
    encoder: Encoder | None = None
    classes: list[str] = field(default_factory=list)
    W: list[list[float]] = field(default_factory=list)
    b: list[float] = field(default_factory=list)
    n_train: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    emulator = LOGISTIC_VERSION

    @property
    def dropped(self) -> tuple[str, ...]:
        return MEMORY_FEATURES if self.feature_set == "no_memory" else ()

    def _encoder(self) -> Encoder:
        numeric = tuple(n for n in NUMERIC if n not in self.dropped)
        boolean = tuple(n for n in BOOLEAN if n not in self.dropped)
        categorical = tuple(n for n in CATEGORICAL if n not in self.dropped)
        return Encoder(numeric, boolean, categorical)

    def fit(self, rows: list[dict]) -> "LogisticPolicy":
        rows = [r for r in rows if r["intended_action"] in self.actions]
        self.encoder = self._encoder().fit([r["features"] for r in rows])
        self.classes = sorted({r["intended_action"] for r in rows}, key=self.actions.index)
        X = [self.encoder.transform(r["features"]) for r in rows]
        y = [self.classes.index(r["intended_action"]) for r in rows]
        weights = None
        if self.class_weight == "balanced":
            counts = Counter(y)
            weights = [len(y) / (len(counts) * counts[label]) for label in y]
        self.W, self.b, self.info = fit_softmax(X, y, len(self.encoder.columns), len(self.classes), c=self.c,
                                                weights=weights)
        self.n_train = len(rows)
        return self

    def raw_distribution(self, features: dict[str, Any]) -> dict[str, float]:
        p = predict_softmax(self.W, self.b, self.encoder.transform(features))
        by_class = dict(zip(self.classes, p))
        floor = self.epsilon / len(self.actions)
        return {a: (1 - self.epsilon) * by_class.get(a, 0.0) + floor for a in self.actions}

    def top_features(self, features: dict[str, Any], action: str, n: int = 4) -> list[tuple[str, float]]:
        """The encoded columns pushing `action` up or down most, relative to the class average."""
        if action not in self.classes:
            return []
        k = self.classes.index(action)
        out = []
        for j, v in self.encoder.transform(features):
            row = self.W[j]
            out.append((self.encoder.columns[j], (row[k] - sum(row) / len(row)) * v))
        return [(c, round(w, 3)) for c, w in sorted(out, key=lambda x: (-abs(x[1]), x[0]))[:n]]

    def predict(self, features: dict[str, Any], legal: list[str]) -> Prediction:
        raw = self.raw_distribution(features)
        probs = mask_and_renormalize(raw, legal)
        top, raw_top = _top(probs), _top(raw)
        return Prediction(raw=raw, probs=probs, top_action=top, raw_top_action=raw_top,
                          masking_changed_top=top != raw_top, matched_bucket="-", backoff_level=0,
                          sample_count=self.n_train,
                          explanation={"top_features": self.top_features(features, top) if top else []})

    def to_json(self) -> dict[str, Any]:
        return {"emulator": self.emulator, "c": self.c, "class_weight": self.class_weight,
                "feature_set": self.feature_set, "epsilon": self.epsilon, "actions": list(self.actions),
                "classes": self.classes, "encoder": self.encoder.to_json(),
                "W": [[round(w, 8) for w in row] for row in self.W], "b": [round(v, 8) for v in self.b],
                "n_train": self.n_train, "info": self.info}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LogisticPolicy":
        return cls(c=data["c"], class_weight=data["class_weight"], feature_set=data["feature_set"],
                   epsilon=data["epsilon"], actions=tuple(data["actions"]), encoder=Encoder.from_json(data["encoder"]),
                   classes=list(data["classes"]), W=[list(r) for r in data["W"]], b=list(data["b"]),
                   n_train=data["n_train"], info=data.get("info") or {})


# -- B3 ------------------------------------------------------------------------------------------

def _split_value(features: dict[str, Any], name: str) -> Any:
    value = features.get(name)
    if name in NUMERIC:
        return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else -1.0
    if name in BOOLEAN:
        return bool(value)
    return str(value)


def _goes_left(features: dict[str, Any], split: dict[str, Any]) -> bool:
    v = _split_value(features, split["feature"])
    if split["kind"] == "numeric":
        return v <= split["threshold"]
    if split["kind"] == "boolean":
        return v is True
    return v == split["value"]


def _condition(split: dict[str, Any], left: bool) -> str:
    f = split["feature"]
    if split["kind"] == "numeric":
        return f"{f} {'<=' if left else '>'} {split['threshold']:g}"
    if split["kind"] == "boolean":
        return f"{f} = {'true' if left else 'false'}"
    return f"{f} {'=' if left else '!='} {split['value']}"


def _gini(counts: Counter, n: int) -> float:
    return 1.0 - sum((c / n) ** 2 for c in counts.values()) if n else 0.0


@dataclass
class TreePolicy:
    max_depth: int = 4
    min_samples_leaf: int = 20
    alpha: float = 2.0
    actions: tuple[str, ...] = schema.ACTIONS
    nodes: list[dict[str, Any]] = field(default_factory=list)
    n_train: int = 0

    emulator = TREE_VERSION
    FEATURES = NUMERIC + BOOLEAN + CATEGORICAL

    def fit(self, rows: list[dict]) -> "TreePolicy":
        rows = [r for r in rows if r["intended_action"] in self.actions]
        self.n_train = len(rows)
        self.nodes = []
        uniform = {a: 1.0 / len(self.actions) for a in self.actions}
        self._grow([(r["features"], r["intended_action"]) for r in rows], 0, uniform, 1.0)
        return self

    def _grow(self, data: list[tuple[dict, str]], depth: int, parent: dict[str, float], alpha: float) -> int:
        counts = Counter(a for _, a in data)
        n = len(data)
        dist = {a: (counts.get(a, 0) + alpha * parent[a]) / (n + alpha) for a in self.actions}
        node = {"id": len(self.nodes), "depth": depth, "n": n, "counts": dict(sorted(counts.items())), "dist": dist}
        self.nodes.append(node)
        split = self._best_split(data, counts) if depth < self.max_depth else None
        if split is not None:
            left = [d for d in data if _goes_left(d[0], split)]
            right = [d for d in data if not _goes_left(d[0], split)]
            node["split"] = split
            node["left"] = self._grow(left, depth + 1, dist, self.alpha)
            node["right"] = self._grow(right, depth + 1, dist, self.alpha)
        return node["id"]

    def _best_split(self, data: list[tuple[dict, str]], counts: Counter) -> dict[str, Any] | None:
        n, m = len(data), self.min_samples_leaf
        if n < 2 * m or len(counts) < 2:
            return None
        parent = _gini(counts, n)
        best, best_gain = None, 1e-9
        for name in self.FEATURES:
            if name in NUMERIC:
                pairs = sorted(((_split_value(f, name), a) for f, a in data), key=lambda x: x[0])
                left, right = Counter(), Counter(a for _, a in pairs)
                for i in range(n - 1):
                    left[pairs[i][1]] += 1
                    right[pairs[i][1]] -= 1
                    if pairs[i][0] == pairs[i + 1][0] or i + 1 < m or n - i - 1 < m:
                        continue
                    nl = i + 1
                    gain = parent - (nl * _gini(left, nl) + (n - nl) * _gini(+right, n - nl)) / n
                    if gain > best_gain:
                        threshold = round((pairs[i][0] + pairs[i + 1][0]) / 2, 6)
                        best, best_gain = {"feature": name, "kind": "numeric", "threshold": threshold}, gain
                continue
            candidates = [True] if name in BOOLEAN else sorted({_split_value(f, name) for f, _ in data})
            for value in candidates:
                split = ({"feature": name, "kind": "boolean"} if name in BOOLEAN
                         else {"feature": name, "kind": "categorical", "value": value})
                left = Counter(a for f, a in data if _goes_left(f, split))
                nl = sum(left.values())
                if nl < m or n - nl < m:
                    continue
                right = counts - left
                gain = parent - (nl * _gini(left, nl) + (n - nl) * _gini(right, n - nl)) / n
                if gain > best_gain:
                    best, best_gain = split, gain
        if best is not None:
            best["gain"] = round(best_gain, 6)
        return best

    def leaf(self, features: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        node, path = self.nodes[0], []
        while "split" in node:
            left = _goes_left(features, node["split"])
            path.append(_condition(node["split"], left))
            node = self.nodes[node["left"] if left else node["right"]]
        return node, path

    def predict(self, features: dict[str, Any], legal: list[str]) -> Prediction:
        node, path = self.leaf(features)
        raw = dict(node["dist"])
        probs = mask_and_renormalize(raw, legal)
        top, raw_top = _top(probs), _top(raw)
        return Prediction(raw=raw, probs=probs, top_action=top, raw_top_action=raw_top,
                          masking_changed_top=top != raw_top, matched_bucket=f"leaf {node['id']}",
                          backoff_level=0, sample_count=node["n"],
                          explanation={"leaf": node["id"], "path": path, "leaf_samples": node["n"],
                                       "leaf_counts": node["counts"]})

    @property
    def depth(self) -> int:
        return max((n["depth"] for n in self.nodes), default=0)

    @property
    def leaves(self) -> list[dict[str, Any]]:
        return [n for n in self.nodes if "split" not in n]

    def rules(self) -> list[dict[str, Any]]:
        """Every leaf as a readable rule (an explanation of the surrogate, not of the LLM)."""
        out = []

        def walk(node, path):
            if "split" not in node:
                top = max(node["dist"], key=node["dist"].get)
                out.append({"leaf": node["id"], "path": path, "n": node["n"], "top": top,
                            "p_top": round(node["dist"][top], 3), "counts": node["counts"]})
                return
            walk(self.nodes[node["left"]], path + [_condition(node["split"], True)])
            walk(self.nodes[node["right"]], path + [_condition(node["split"], False)])
        walk(self.nodes[0], [])
        return out

    def to_json(self) -> dict[str, Any]:
        return {"emulator": self.emulator, "max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf,
                "alpha": self.alpha, "actions": list(self.actions), "n_train": self.n_train,
                "nodes": [{**n, "dist": {a: round(p, 8) for a, p in n["dist"].items()}} for n in self.nodes]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TreePolicy":
        return cls(max_depth=data["max_depth"], min_samples_leaf=data["min_samples_leaf"], alpha=data["alpha"],
                   actions=tuple(data["actions"]), nodes=[dict(n) for n in data["nodes"]], n_train=data["n_train"])


def entropy_bits(probs: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)
