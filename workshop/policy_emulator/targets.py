"""Target Policy v2: score each legal target from observable preference features.

Only MOVE offers a real choice of target in the live payload (interact,
inspect, place_trap, disarm and enter_safe_code act on the one object in
range; approach / avoid on the opponent). For MOVE, v2 is a conditional
logit over the legal move targets:

    P(target) proportional to exp(beta . phi(target))

fitted per policy identity (L2, pure-Python L-BFGS). phi uses only what the
model sees in its payload (`target_features`, versioned target-features-v1):

* the v1 target class (exit / door untraversed / traversed / blocked / safe /
  unsearched here / unsearched elsewhere / other object);
* the MOVE ETA the payload quotes (`action_costs.move_eta_s`; null means the
  route crosses unexplored rooms - the LLM sees the same null), nearest in its
  class, nearest overall;
* the door is an exit of the current room; the object is listed as inspected
  / searched in `room.objects`; the object is the one in range;
* the exit when the mission is complete; the safe when its code is known or
  can be worked out from what was read.

Every other action keeps the v1 behaviour (a single legal target).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from workshop.policy_emulator import schema
from workshop.policy_emulator.linear import lbfgs, softmax
from workshop.policy_emulator.policies import TargetChoice, TargetPolicy, bucket_key

TARGET_FEATURES_VERSION = "target-features-v1"
TARGET_V2_VERSION = "target-v2"
TARGET_V21_VERSION = "target-v2.1"
MOVE_CLASSES = ("exit_door", "door_untraversed", "door_traversed", "door_blocked", "safe", "unsearched_here",
                "unsearched_elsewhere", "other_object")
FEATURE_NAMES = tuple(f"class={c}" for c in MOVE_CLASSES) + (
    "eta_known", "eta_10s", "nearest_in_class", "nearest_overall", "door_here", "listed_inspected", "listed_searched",
    "near_object", "exit_and_mission_complete", "safe_and_code_known")


def target_features(payload: dict[str, Any], targets: list[str]) -> dict[str, dict[str, float]]:
    """phi for every candidate MOVE target (observable payload fields only)."""
    etas = ((payload.get("action_costs") or {}).get("move_eta_s") or {})
    objects = {o["id"]: o for o in (payload.get("room") or {}).get("objects") or [] if isinstance(o, dict) and o.get("id")}
    exploration = payload.get("exploration") or {}
    here = exploration.get("current_room") or (payload.get("self") or {}).get("room")
    room_facts = (exploration.get("rooms") or {}).get(here) or {}
    doors_here = set(room_facts.get("traversed_exits") or {}) | set(room_facts.get("untraversed_exits") or []) \
        | set(room_facts.get("blocked") or {})
    near = (payload.get("self") or {}).get("near_object_id")
    complete = bool((payload.get("mission") or {}).get("ready_to_exit"))
    code_known = bool((payload.get("puzzle") or {}).get("safe_code_known")) or schema.code_from_visible_puzzle(payload) is not None
    classes = {t: schema.target_class("move", t, payload) for t in targets}
    known = {t: float(etas[t]) for t in targets if isinstance(etas.get(t), int | float)}
    best_overall = min(known.values()) if known else None
    best_class: dict[str, float] = {}
    for t, eta in known.items():
        c = classes[t]
        best_class[c] = min(best_class.get(c, eta), eta)
    out = {}
    for t in targets:
        c = classes[t]
        eta = known.get(t)
        obj = objects.get(t) or {}
        phi = {f"class={c}": 1.0}
        if eta is not None:
            phi["eta_known"] = 1.0
            phi["eta_10s"] = min(eta, 60.0) / 10.0
            if eta == best_class.get(c):
                phi["nearest_in_class"] = 1.0
            if eta == best_overall:
                phi["nearest_overall"] = 1.0
        if t in doors_here:
            phi["door_here"] = 1.0
        if obj.get("inspected"):
            phi["listed_inspected"] = 1.0
        if obj.get("searched"):
            phi["listed_searched"] = 1.0
        if t == near:
            phi["near_object"] = 1.0
        if c == "exit_door" and complete:
            phi["exit_and_mission_complete"] = 1.0
        if c == "safe" and code_known:
            phi["safe_and_code_known"] = 1.0
        out[t] = phi
    return out


def _move_targets(payload: dict[str, Any]) -> list[str]:
    return [t for t in schema.legal_targets(payload).get("move", []) if t]


@dataclass
class TargetPolicyV2:
    """v2 for MOVE (conditional logit); a v1 TargetPolicy for everything else and for trap types."""
    c: float = 1.0
    beta: dict[str, float] = field(default_factory=dict)
    v1: TargetPolicy = field(default_factory=TargetPolicy)
    n_train: int = 0
    info: dict[str, Any] = field(default_factory=dict)
    # "zero": L2 towards 0 (v2). "pooled": L2 towards the fit on all policies' training rows
    # (pooled_rows), so a coefficient with little support for this identity stays near the
    # shared estimate instead of near 0.
    prior: str = "zero"

    version = TARGET_V2_VERSION
    feature_names = FEATURE_NAMES

    def phis(self, payload: dict[str, Any], targets: list[str], features: dict[str, Any] | None = None
             ) -> dict[str, dict[str, float]]:
        return target_features(payload, targets)

    def _cases(self, rows: list[dict], index: dict[str, int]) -> list:
        sparse = []
        for r in rows:
            if r["intended_action"] != "move" or not r.get("intended_target"):
                continue
            targets = _move_targets(r["observation"])
            if len(targets) < 2 or r["intended_target"] not in targets:
                continue
            phis = self.phis(r["observation"], targets, r["features"])
            sparse.append(([[(index[k], v) for k, v in sorted(phis[t].items())] for t in targets],
                           targets.index(r["intended_target"])))
        return sparse

    def _solve(self, sparse: list, names: list[str], prior: list[float]) -> tuple[list[float], dict]:
        inv = 1.0 / self.c

        def fg(beta: list[float]) -> tuple[float, list[float]]:
            loss, grad = 0.0, [0.0] * len(names)
            for cands, y in sparse:
                scores = [sum(beta[j] * v for j, v in phi) for phi in cands]
                p = softmax(scores)
                loss -= math.log(max(p[y], 1e-300))
                for phi, pk in zip(cands, p):
                    for j, v in phi:
                        grad[j] += pk * v
                for j, v in cands[y]:
                    grad[j] -= v
            for i, b in enumerate(beta):
                loss += 0.5 * inv * (b - prior[i]) ** 2
                grad[i] += inv * (b - prior[i])
            return loss, grad
        return lbfgs(fg, list(prior))

    def fit(self, rows: list[dict], pooled_rows: list[dict] | None = None) -> "TargetPolicyV2":
        self.v1 = TargetPolicy().fit(rows, pooled_rows)
        names = list(self.feature_names)
        index = {n: i for i, n in enumerate(names)}
        prior = [0.0] * len(names)
        if self.prior == "pooled" and pooled_rows:
            prior, _ = self._pooled_prior(pooled_rows, names, index)
        sparse = self._cases(rows, index)
        if sparse:
            theta, self.info = self._solve(sparse, names, prior)
            self.beta = {n: round(theta[i], 8) for i, n in enumerate(names)}
        self.n_train = len(sparse)
        return self

    _POOLED_CACHE: Any = None  # (signature, prior) - the pooled fit is shared by the identities of one fold

    def _pooled_prior(self, pooled_rows: list[dict], names: list[str], index: dict[str, int]) -> tuple[list[float], dict]:
        signature = (type(self).__name__, self.c, len(pooled_rows), pooled_rows[0].get("decision_id"),
                     pooled_rows[-1].get("decision_id"))
        cached = TargetPolicyV2._POOLED_CACHE
        if cached and cached[0] == signature:
            return cached[1]
        result = self._solve(self._cases(pooled_rows, index), names, [0.0] * len(names))
        TargetPolicyV2._POOLED_CACHE = (signature, result)
        return result

    def move_distribution(self, payload: dict[str, Any], features: dict[str, Any] | None = None) -> dict[str, float]:
        targets = _move_targets(payload)
        if not targets:
            return {}
        phis = self.phis(payload, targets, features)
        scores = [sum(self.beta.get(k, 0.0) * v for k, v in phis[t].items()) for t in targets]
        return dict(zip(targets, softmax(scores)))

    def choose(self, action: str, payload: dict[str, Any], features: dict[str, Any]) -> TargetChoice:
        if action != "move" or not self.beta:
            return self.v1.choose(action, payload, features)
        dist = self.move_distribution(payload, features)
        if len(dist) <= 1:
            return self.v1.choose(action, payload, features)
        classes: dict[str, float] = defaultdict(float)
        for t, p in dist.items():
            classes[schema.target_class("move", t, payload)] += p
        return TargetChoice(dist, dict(classes), f"{self.version}:move", False)

    def trap_type(self, payload: dict[str, Any], rng: random.Random) -> str | None:
        return self.v1.trap_type(payload, rng)

    def to_json(self) -> dict[str, Any]:
        return {"emulator": self.version, "target_features_version": TARGET_FEATURES_VERSION, "c": self.c,
                "beta": self.beta, "n_train": self.n_train, "info": self.info, "v1": self.v1.to_json(),
                **({"prior": self.prior} if self.prior != "zero" else {}),
                "form": "conditional logit over the legal MOVE targets; other actions as target-class-v1"}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TargetPolicyV2":
        return cls(c=data["c"], beta=dict(data["beta"]), v1=TargetPolicy.from_json(data["v1"]),
                   n_train=data["n_train"], info=data.get("info") or {}, prior=data.get("prior", "zero"))


CONTEXT_DIMENSIONS = ("mission_complete", "opponent_alert_present")  # = v1's TARGET_DIMENSIONS
CONTEXT_FEATURE_NAMES = tuple(f"class={c}&{d}" for d in CONTEXT_DIMENSIONS for c in MOVE_CLASSES)


@dataclass
class TargetPolicyV21(TargetPolicyV2):
    """target-v2.1: target-v2 plus target class x context interactions.

    v2 scores every legal MOVE target with one weight vector, so the mission
    state reaches it only through `exit_and_mission_complete`, one coefficient
    fitted on 2.5% of the training decisions and shrunk by the L2 penalty; the
    other targets keep their usual attraction when the mission is complete.
    v2.1 adds one indicator per (target class, context flag) for the two
    observable context flags v1 conditions on (mission_complete,
    opponent_alert_present), so each class's attraction can change with the
    mission state and with an alert. Same fit (conditional logit, L2, C=1)."""
    prior: str = "pooled"

    version = TARGET_V21_VERSION
    feature_names = FEATURE_NAMES + CONTEXT_FEATURE_NAMES

    def phis(self, payload: dict[str, Any], targets: list[str], features: dict[str, Any] | None = None
             ) -> dict[str, dict[str, float]]:
        base = target_features(payload, targets)
        flags = [d for d in CONTEXT_DIMENSIONS if (features or {}).get(d)]
        if not flags:
            return base
        out = {}
        for t, phi in base.items():
            cls = next(k[len("class="):] for k in phi if k.startswith("class="))
            out[t] = {**phi, **{f"class={cls}&{d}": 1.0 for d in flags}}
        return out

    def to_json(self) -> dict[str, Any]:
        return {**super().to_json(), "context_dimensions": list(CONTEXT_DIMENSIONS),
                "form": "conditional logit over the legal MOVE targets with target class x context "
                        "(mission_complete, opponent_alert_present) indicators; other actions as target-class-v1"}


WITHIN_CLASS_FEATURES = tuple(n for n in FEATURE_NAMES if not n.startswith("class="))


@dataclass
class FactorizedTargetPolicy:
    """Diagnostic alternative to target-v2.1 (evaluated, not shipped):
    P(target) = P(class | context) x P(target | class).

    The class step is the v1 table (target class per action and context bucket
    mission_complete x opponent_alert_present, with v1's backoff), plus a small
    pseudo-count `class_alpha` on every legal class so an unseen class is unlikely,
    not impossible. The within-class step is a conditional logit over the legal
    targets of that class on the observable v2 features minus the class indicators."""
    c: float = 1.0
    class_alpha: float = 0.05
    beta: dict[str, float] = field(default_factory=dict)
    v1: TargetPolicy = field(default_factory=TargetPolicy)
    n_train: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    version = "target-v2-factorized"

    def fit(self, rows: list[dict], pooled_rows: list[dict] | None = None) -> "FactorizedTargetPolicy":
        self.v1 = TargetPolicy().fit(rows, pooled_rows)
        names = list(WITHIN_CLASS_FEATURES)
        index = {n: i for i, n in enumerate(names)}
        sparse = []
        for r in rows:
            if r["intended_action"] != "move" or not r.get("intended_target"):
                continue
            payload = r["observation"]
            targets = _move_targets(payload)
            if r["intended_target"] not in targets:
                continue
            real_class = schema.target_class("move", r["intended_target"], payload)
            members = [t for t in targets if schema.target_class("move", t, payload) == real_class]
            if len(members) < 2:
                continue
            phis = target_features(payload, members)
            sparse.append(([[(index[k], v) for k, v in sorted(phis[t].items()) if k in index] for t in members],
                           members.index(r["intended_target"])))
        inv = 1.0 / self.c

        def fg(beta: list[float]) -> tuple[float, list[float]]:
            loss, grad = 0.0, [0.0] * len(names)
            for cands, y in sparse:
                p = softmax([sum(beta[j] * v for j, v in phi) for phi in cands])
                loss -= math.log(max(p[y], 1e-300))
                for phi, pk in zip(cands, p):
                    for j, v in phi:
                        grad[j] += pk * v
                for j, v in cands[y]:
                    grad[j] -= v
            for i, b in enumerate(beta):
                loss += 0.5 * inv * b * b
                grad[i] += inv * b
            return loss, grad
        if sparse:
            theta, self.info = lbfgs(fg, [0.0] * len(names))
            self.beta = {n: round(theta[i], 8) for i, n in enumerate(names)}
        self.n_train = len(sparse)
        return self

    def class_distribution(self, payload: dict[str, Any], features: dict[str, Any],
                           classes: dict[str, list[str]]) -> tuple[dict[str, float], str]:
        """v1's backoff over the context tables, smoothed over the legal classes."""
        key = f"move|{bucket_key(features, self.v1.dims)}"
        for source, counts, need in ((f"policy:{key}", self.v1.tables.get("exact", {}).get(key), self.v1.min_count),
                                     ("policy:move", self.v1.tables.get("action", {}).get("move"), 1),
                                     ("pooled:move", self.v1.pooled.get("move"), 1)):
            present = {c: n for c, n in (counts or {}).items() if c in classes}
            if sum(present.values()) >= need:
                weights = {c: present.get(c, 0) + self.class_alpha for c in classes}
                total = sum(weights.values())
                return {c: w / total for c, w in weights.items()}, source
        return {c: 1.0 / len(classes) for c in classes}, "uniform"

    def move_distribution(self, payload: dict[str, Any], features: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], str]:
        targets = _move_targets(payload)
        if not targets:
            return {}, {}, "none"
        classes: dict[str, list[str]] = defaultdict(list)
        for t in targets:
            classes[schema.target_class("move", t, payload)].append(t)
        class_probs, source = self.class_distribution(payload, features, classes)
        phis = target_features(payload, targets)
        dist: dict[str, float] = {}
        for c, members in classes.items():
            within = softmax([sum(self.beta.get(k, 0.0) * v for k, v in phis[t].items()) for t in members])
            for t, p in zip(members, within):
                dist[t] = class_probs[c] * p
        return dist, class_probs, source

    def choose(self, action: str, payload: dict[str, Any], features: dict[str, Any]) -> TargetChoice:
        if action != "move" or not self.beta:
            return self.v1.choose(action, payload, features)
        dist, class_probs, source = self.move_distribution(payload, features)
        if len(dist) <= 1:
            return self.v1.choose(action, payload, features)
        return TargetChoice(dist, class_probs, f"{self.version}:{source}", source == "uniform")

    def trap_type(self, payload: dict[str, Any], rng: random.Random) -> str | None:
        return self.v1.trap_type(payload, rng)

    def to_json(self) -> dict[str, Any]:
        return {"emulator": self.version, "target_features_version": TARGET_FEATURES_VERSION, "c": self.c,
                "class_alpha": self.class_alpha, "beta": self.beta, "n_train": self.n_train, "info": self.info,
                "v1": self.v1.to_json(),
                "form": "P(class | move, context) from the v1 tables (+class_alpha on legal classes) x "
                        "conditional logit within the class; other actions as target-class-v1"}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "FactorizedTargetPolicy":
        return cls(c=data["c"], class_alpha=data["class_alpha"], beta=dict(data["beta"]),
                   v1=TargetPolicy.from_json(data["v1"]), n_train=data["n_train"], info=data.get("info") or {})


def target_metrics(policy, rows: list[dict]) -> dict[str, Any]:
    """Target fidelity given the LLM's real action (v1 or v2 policy)."""
    out: dict[str, Any] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        action, real = r["intended_action"], r.get("intended_target")
        if action is None:
            continue
        payload = r["observation"]
        legal = [t for t in schema.legal_targets(payload).get(action, []) if t is not None]
        choice = policy.choose(action, payload, r["features"])
        top = max(choice.distribution, key=lambda t: (choice.distribution[t], -legal.index(t) if t in legal else 0)) \
            if choice.distribution else None
        p_real = choice.distribution.get(real, 0.0)
        rec = {"action": action, "n_targets": len(legal), "p_real": p_real, "exact": top == real,
               "class_top": (max(choice.target_class_probs, key=choice.target_class_probs.get)
                             if choice.target_class_probs else None) == r.get("target_class"),
               "uniform": 1.0 / len(legal) if legal else 1.0, "legal": all(t in legal or t is None for t in choice.distribution),
               "real_class": r.get("target_class"), "policy_key": r["policy_key"]}
        groups["all"].append(rec)
        groups[f"action:{action}"].append(rec)
        if len(legal) > 1:
            groups["multi"].append(rec)
            groups[f"class:{r.get('target_class')}"].append(rec)
            groups[f"policy:{r['policy_key']}"].append(rec)

    def block(recs):
        n = len(recs)
        if not n:
            return {"n": 0}
        return {"n": n, "class_top1": round(sum(x["class_top"] for x in recs) / n, 4),
                "exact_top1": round(sum(x["exact"] for x in recs) / n, 4),
                "mean_p_real": round(sum(x["p_real"] for x in recs) / n, 4),
                "log_loss": round(-sum(math.log(max(x["p_real"], 1e-6)) for x in recs) / n, 4),
                "uniform_baseline": round(sum(x["uniform"] for x in recs) / n, 4),
                "legal_target_rate": round(sum(x["legal"] for x in recs) / n, 4),
                "single_target": sum(1 for x in recs if x["n_targets"] <= 1)}
    for key, recs in sorted(groups.items()):
        out[key] = block(recs)
    return out
