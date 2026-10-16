"""Pure-Python numerics for Stage 3 milestone 2 (no numpy in this project).

* `Encoder` - deterministic preprocessing of PolicyFeatureExtractorV1 features:
  booleans as 0/1, numerics standardized with the training mean / std (an
  explicit missing indicator where a value can be null), categoricals one-hot
  with a stable "__other__" column for values never seen in training.
* `lbfgs` - a small L-BFGS minimizer (two-loop recursion, backtracking Armijo
  line search), deterministic.
* `fit_softmax` - L2-regularized multinomial logistic regression on sparse
  rows (list of (column, value)), with optional per-row weights.

Everything is JSON-serializable; prediction needs no fitting at runtime.
"""
from __future__ import annotations

import math
from typing import Any, Callable

OTHER = "__other__"
MISSING = "__missing__"

# PolicyFeatureExtractorV1 features used by B2 / B3 (identity-like or redundant ones left out:
# time_remaining_seconds duplicates the fraction; last_action_type == recent_action_1)
NUMERIC = ("time_remaining_fraction", "objectives_remaining_count", "hearts", "trap_hits", "trap_inventory_total",
           "own_traps_armed", "legal_action_count", "move_target_count", "room_unsearched_count",
           "room_uninspected_count", "room_inspected_unsearched_count", "room_untraversed_exits_count",
           "known_unsearched_elsewhere_count", "visited_room_count", "known_locked_object_count", "opponent_hearts")
BOOLEAN = ("has_archive_key", "has_security_keycard", "has_safe_code", "safe_code_derivable", "has_secret_documents",
           "has_passport", "mission_complete", "exit_known", "exit_reachable_if_observable", "opponent_visible",
           "opponent_alert_present", "opponent_same_room", "fight_possible_now", "inspect_legal", "search_legal",
           "door_interact_legal", "place_trap_legal", "disarm_legal", "approach_opponent_legal", "avoid_opponent_legal",
           "enter_code_legal", "exit_interact_legal", "fight_related_option_present", "recent_trap_attempt",
           "opponent_seen_before")
CATEGORICAL = ("time_pressure", "health_band", "life_balance", "opponent_distance_bucket", "room_status",
               "near_object_type", "current_room", "recent_action_1", "recent_action_2")


class Encoder:
    """Fitted once on training rows; then a pure function of the features."""

    def __init__(self, numeric=NUMERIC, boolean=BOOLEAN, categorical=CATEGORICAL) -> None:
        self.numeric, self.boolean, self.categorical = tuple(numeric), tuple(boolean), tuple(categorical)
        self.stats: dict[str, tuple[float, float]] = {}
        self.categories: dict[str, list[str]] = {}
        self.columns: list[str] = []

    def fit(self, rows: list[dict]) -> "Encoder":
        for name in self.numeric:
            vals = [float(r[name]) for r in rows if isinstance(r.get(name), int | float) and not isinstance(r.get(name), bool)]
            mean = sum(vals) / len(vals) if vals else 0.0
            var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
            self.stats[name] = (round(mean, 10), round(math.sqrt(var), 10) or 1.0)
        for name in self.categorical:
            self.categories[name] = sorted({str(r.get(name, MISSING)) for r in rows})
        cols = []
        for name in self.numeric:
            cols.append(name)
            if any(r.get(name) is None for r in rows):
                cols.append(f"{name}{MISSING}")
        cols += list(self.boolean)
        for name in self.categorical:
            cols += [f"{name}={v}" for v in self.categories[name]] + [f"{name}={OTHER}"]
        self.columns = cols
        self._index = {c: i for i, c in enumerate(cols)}
        return self

    def transform(self, features: dict[str, Any]) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        idx = self._index
        for name in self.numeric:
            value = features.get(name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                mean, std = self.stats[name]
                z = (float(value) - mean) / std
                if z:
                    out.append((idx[name], z))
            elif f"{name}{MISSING}" in idx:
                out.append((idx[f"{name}{MISSING}"], 1.0))
        for name in self.boolean:
            if features.get(name):
                out.append((idx[name], 1.0))
        for name in self.categorical:
            col = f"{name}={features.get(name, MISSING)}"
            out.append((idx.get(col, idx[f"{name}={OTHER}"]), 1.0))
        return out

    def unseen(self, features: dict[str, Any]) -> list[str]:
        """Categorical values never seen in training (they map to __other__)."""
        return [f"{n}={features.get(n, MISSING)}" for n in self.categorical
                if str(features.get(n, MISSING)) not in self.categories.get(n, [])]

    def to_json(self) -> dict[str, Any]:
        return {"numeric": list(self.numeric), "boolean": list(self.boolean), "categorical": list(self.categorical),
                "stats": {k: list(v) for k, v in self.stats.items()}, "categories": self.categories,
                "columns": self.columns}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Encoder":
        enc = cls(data["numeric"], data["boolean"], data["categorical"])
        enc.stats = {k: (v[0], v[1]) for k, v in data["stats"].items()}
        enc.categories = {k: list(v) for k, v in data["categories"].items()}
        enc.columns = list(data["columns"])
        enc._index = {c: i for i, c in enumerate(enc.columns)}
        return enc


def lbfgs(fg: Callable[[list[float]], tuple[float, list[float]]], x0: list[float], *, max_iter: int = 200,
          memory: int = 8, tol: float = 1e-6) -> tuple[list[float], dict[str, Any]]:
    """Minimizes f; fg returns (f, gradient). Deterministic."""
    x = list(x0)
    f, g = fg(x)
    s_hist: list[list[float]] = []
    y_hist: list[list[float]] = []
    n = len(x)
    it = 0
    for it in range(1, max_iter + 1):
        # two-loop recursion: d = -H g
        q = list(g)
        alphas = []
        for s, y in zip(reversed(s_hist), reversed(y_hist)):
            rho = 1.0 / sum(a * b for a, b in zip(y, s))
            a = rho * sum(si * qi for si, qi in zip(s, q))
            alphas.append((rho, a))
            q = [qi - a * yi for qi, yi in zip(q, y)]
        if y_hist:
            s, y = s_hist[-1], y_hist[-1]
            gamma = sum(a * b for a, b in zip(s, y)) / sum(b * b for b in y)
        else:
            gamma = 1.0 / max(1e-12, math.sqrt(sum(v * v for v in g)))
        r = [gamma * v for v in q]
        for (s, y), (rho, a) in zip(zip(s_hist, y_hist), reversed(alphas)):
            b = rho * sum(yi * ri for yi, ri in zip(y, r))
            r = [ri + si * (a - b) for ri, si in zip(r, s)]
        d = [-v for v in r]
        slope = sum(a * b for a, b in zip(g, d))
        if slope >= 0:  # not a descent direction: restart from steepest descent
            s_hist, y_hist = [], []
            d = [-v for v in g]
            slope = -sum(v * v for v in g)
        step = 1.0
        while True:
            x_new = [xi + step * di for xi, di in zip(x, d)]
            f_new, g_new = fg(x_new)
            if f_new <= f + 1e-4 * step * slope or step < 1e-10:
                break
            step *= 0.5
        s = [a - b for a, b in zip(x_new, x)]
        y = [a - b for a, b in zip(g_new, g)]
        sy = sum(a * b for a, b in zip(s, y))
        if sy > 1e-12:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > memory:
                s_hist.pop(0)
                y_hist.pop(0)
        converged = abs(f - f_new) <= tol * max(1.0, abs(f))
        x, f, g = x_new, f_new, g_new
        if converged or math.sqrt(sum(v * v for v in g)) < tol:
            break
    return x, {"iterations": it, "objective": f, "n_params": n}


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    s = sum(e)
    return [v / s for v in e]


def fit_softmax(rows: list[list[tuple[int, float]]], labels: list[int], n_cols: int, n_classes: int, *,
                c: float = 1.0, weights: list[float] | None = None, max_iter: int = 200) -> tuple[list[list[float]], list[float], dict]:
    """L2 multinomial logistic regression. Objective: sum_i w_i * NLL_i + ||W||^2 / (2c)
    (intercepts unpenalized). Returns (W[col][class], b[class], info)."""
    weights = weights or [1.0] * len(rows)
    k = n_classes
    n_params = n_cols * k + k

    def fg(theta: list[float]) -> tuple[float, list[float]]:
        grad = [0.0] * n_params
        loss = 0.0
        base = n_cols * k
        for x, y, w in zip(rows, labels, weights):
            logits = theta[base:base + k]
            logits = list(logits)
            for j, v in x:
                off = j * k
                for c_ in range(k):
                    logits[c_] += theta[off + c_] * v
            p = softmax(logits)
            loss -= w * math.log(max(p[y], 1e-300))
            p[y] -= 1.0
            for c_ in range(k):
                grad[base + c_] += w * p[c_]
            for j, v in x:
                off = j * k
                wv = w * v
                for c_ in range(k):
                    grad[off + c_] += wv * p[c_]
        inv = 1.0 / c
        for i in range(base):
            loss += 0.5 * inv * theta[i] * theta[i]
            grad[i] += inv * theta[i]
        return loss, grad

    theta, info = lbfgs(fg, [0.0] * n_params, max_iter=max_iter)
    W = [theta[j * k:(j + 1) * k] for j in range(n_cols)]
    return W, theta[n_cols * k:], info


def predict_softmax(W: list[list[float]], b: list[float], x: list[tuple[int, float]]) -> list[float]:
    logits = list(b)
    for j, v in x:
        row = W[j]
        for c_ in range(len(logits)):
            logits[c_] += row[c_] * v
    return softmax(logits)
