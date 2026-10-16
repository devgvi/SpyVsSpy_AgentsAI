"""Stage 5A - offline B2-v2 candidate training, DEVELOPMENT selection, freeze and the Stage-4C regression guard
(Archive/STAGE5A_DATA_PROMOTION_AND_OFFLINE_RETRAINING.md sections 10-23, 35).

Order (each artifact written once, never rewritten):
    workshop/policies/stage5/stage5a_selection_protocol.json  candidates, margins, escalation definition, decision rule:
                                                              frozen before any candidate is trained or evaluated
    workshop/policies/stage5/candidates/                      C0 / C1 / C2w2 / C2w4, one logistic-v1 per identity
    workshop/policies/stage5/stage5a_development.json         DEVELOPMENT fidelity, replayed R2.1 escalation, selection
    workshop/policies/stage5/b2-v2-candidate/                 the selected candidate, frozen with a hashed manifest
    workshop/policies/stage5/stage5a_regression.json          Stage-4C regression-only guard and the decision class

Stage 4C is read by `regression` only, after the freeze manifest verifies. `train` and `select` run under a file-open
trace (TRACE) and stop if any Stage-4C data path was opened. Nothing here calls a provider: B2-v1, B1 and R2.1 are
the frozen local artifacts and every escalation figure is an analytic replay of the frozen R2.1 rules.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from workshop.policy_emulator import dataset as ds
from workshop.policy_emulator import router as rt
from workshop.policy_emulator import schema
from workshop.policy_emulator import stage5a_promotion as pr
from workshop.policy_emulator.artifacts import ARTIFACT_VERSION, _load, _write, content_hash, load_policy
from workshop.policy_emulator.evaluation import _metrics, opportunity_rates
from workshop.policy_emulator.features import FEATURE_SCHEMA_VERSION, feature_schema
from workshop.policy_emulator.linear import fit_softmax
from workshop.policy_emulator.models import LogisticPolicy
from workshop.runners.common import PROJECT_ROOT, WorkshopError

PROTOCOL_PATH = pr.STAGE5_DIR / "stage5a_selection_protocol.json"
CANDIDATE_ROOT = pr.STAGE5_DIR / "candidates"
TRAINING_RECORD = CANDIDATE_ROOT / "training_record.json"
DEVELOPMENT_PATH = pr.STAGE5_DIR / "stage5a_development.json"
FREEZE_DIR = pr.STAGE5_DIR / "b2-v2-candidate"
FREEZE_MANIFEST = FREEZE_DIR / "manifest.json"
REGRESSION_PATH = pr.STAGE5_DIR / "stage5a_regression.json"
STAGE4C_REAL_EXPORT = "workshop_runs/stage4c-confirmatory/real_reference"
EMULATOR = "logistic-v1"
B1_EMULATOR = "table-v1"
TARGET = "target-v2.1"
CONFIG = {"c": 1.0, "class_weight": "none", "feature_set": "full"}  # the frozen B2-v1 configuration (model_configs.json)
R21_HASH = "sha256:216f3c9a3c4b3f78"
DOMAINS = {"level_01": "m4d_real", "shifted": "stage4b_real"}
REPLAY_DOMAINS = {"level_01": "m4d", "shifted": "stage4b"}
CODE_FILES = ("workshop/policy_emulator/stage5a_training.py", "workshop/policy_emulator/stage5a_promotion.py",
              "workshop/policy_emulator/models.py", "workshop/policy_emulator/linear.py",
              "workshop/policy_emulator/features.py", "workshop/policy_emulator/evaluation.py",
              "workshop/policy_emulator/router.py")

CANDIDATES: dict[str, dict[str, Any]] = {
    "C0": {"kind": "control", "hard_examples": False, "hard_weight": None,
           "definition": "control retrain: Phase-A CLEAN only (the B2-v1 training rows), logistic-v1 with the B2-v1 "
                         "configuration, features and regularization; must reproduce B2-v1"},
    "C1": {"kind": "candidate", "hard_examples": True, "hard_weight": 1.0,
           "definition": "Phase-A CLEAN + every promoted TRAIN hard example (status kept), unweighted (the 1x point of "
                         "the C2 grid)"},
    "C2w2": {"kind": "candidate", "hard_examples": True, "hard_weight": 2.0,
             "definition": "as C1, each hard example weighted 2x, weights rescaled per identity to mean 1"},
    "C2w4": {"kind": "candidate", "hard_examples": True, "hard_weight": 4.0,
             "definition": "as C1, each hard example weighted 4x, weights rescaled per identity to mean 1"},
}
SIMPLICITY_ORDER = ("C1", "C2w2", "C2w4")
NON_INFERIORITY = {"log_loss": ("lower", 0.02), "top1": ("higher", 0.01), "macro_f1": ("higher", 0.02),
                   "brier": ("lower", 0.01), "jsd": ("lower", 0.005), "opportunity_mae": ("lower", 0.01)}
MATERIAL = {"absolute": 0.01, "relative": 0.10}
TIE_MARGIN = 0.005
FIDELITY_IMPROVEMENT = 0.01
GUARD = {"log_loss_max_increase": 0.05, "top1_max_decrease": 0.03, "opportunity_mae_max_increase": 0.02,
         "expected_escalation": "candidate pooled <= B2-v1 pooled"}

PROTOCOL: dict[str, Any] = {
    "version": "stage5a-selection-protocol-v1",
    "spec": "Archive/STAGE5A_DATA_PROMOTION_AND_OFFLINE_RETRAINING.md",
    "frozen_before": "any candidate is trained and any development metric is computed",
    "model_family": {"emulator": EMULATOR, "config": CONFIG, "epsilon": 1e-3,
                     "identity": "one model per policy identity (model:personality), as B2-v1; no pooled-identity model",
                     "features": "policy-features-v1 through linear.Encoder (the B2-v1 numeric / boolean / categorical "
                                 "columns), unchanged; the encoder is fitted on each candidate's own training rows",
                     "optimizer": "linear.fit_softmax (L-BFGS, max_iter 200, tol 1e-6), unchanged; per-row weights only"},
    "target_policy": "target-v2.1 frozen, not retrained",
    "label": "the REAL action (intended_action); never a router outcome (no router-aware target)",
    "candidates": CANDIDATES,
    "c2_grid": {"1x": "C1", "2x": "C2w2", "4x": "C2w4"},
    "c3": "not trained: no interaction is justified a priori by the existing feature structure (the recent-action, "
          "coverage-like and mission-progress signals are already columns); no feature mining",
    "weighting": "Phase-A rows weight 1, hard examples weight w; per identity every weight is multiplied by "
                 "n_rows / sum(weights) (mean 1), so the L2 penalty keeps its strength relative to the data",
    "c0_reproduction": {"rule": "per identity, the C0 model.json content hash equals the frozen B2-v1 hash, or the largest "
                                "|p_C0 - p_B2v1| over every Phase-A CLEAN and DEV REAL row is <= 1e-6",
                        "on_failure": "STOP: the training pipeline does not reproduce B2-v1"},
    "development": {
        "fidelity_domains": {"level_01": "M4D REAL CLEAN decisions of seeds 8003 / 8004 (Level 1)",
                             "shifted": "Stage-4B REAL CLEAN decisions of seed 9008 (Levels 2 / 3)"},
        "fidelity_metrics": ["log_loss", "top1", "macro_f1", "brier", "jsd", "opportunity_mae"],
        "masking": "every distribution masked to the recorded legal actions (the Stage-3 offline-evaluation convention)",
        "escalation_states": "every decision of the B2-v1-through-R0 matches of the DEVELOPMENT configurations "
                             "(M4D r0_control 8003 / 8004, Stage-4B r0 9008)",
        "hard_states": "the promoted DEVELOPMENT hard examples (states B2-v1 escalated in the DEV Hybrid matches)"},
    "escalation": {
        "primary": "expected R2.1 escalation rate on the DEV R0 replay states, pooled over both domains",
        "per_state": "executable = legal minus unsupported (enter_safe_code without a visible code); B1 = frozen "
                     "table-v1 and the candidate both masked to it; D = round(TVD(B1, candidate), 6) > 0.5; the "
                     "committed history is the logged one; a = the action of the last 4 committed decisions when all "
                     "4 are equal, else none; P_rep = P_candidate(a) (the repetition trigger fires when the sampled "
                     "proposal repeats a); P(escalate) = 1 if D else P_rep",
        "trigger_breakdown": "both = D * P_rep; disagreement only = D * (1 - P_rep); repetition only = (1 - D) * P_rep",
        "router": f"R2.1 frozen ({R21_HASH}): repetition window 4, TVD threshold 0.5; not changed",
        "replay_check": "B2-v1 through the same replay must reproduce every logged disagreement signal, logged TVD and "
                        "logged repetition signal of the R0 states (else STOP)",
        "limitation": "open loop: the states and the history are B2-v1's; a candidate changes its own trajectory only in "
                      "a closed loop (Stage 5B)"},
    "non_inferiority": {k: {"direction": d, "margin": m} for k, (d, m) in NON_INFERIORITY.items()},
    "non_inferiority_rule": "non-inferior when, in each development domain, every metric is within its margin of B2-v1 "
                            "(candidate - B2-v1 <= margin when lower is better, B2-v1 - candidate <= margin when higher "
                            "is better), on the 4-decimal values",
    "material_reduction": {**MATERIAL, "rule": "B2-v1 - candidate >= 0.01 AND (B2-v1 - candidate) / B2-v1 >= 0.10 on "
                                               "the primary expected escalation rate"},
    "tie_margin": TIE_MARGIN,
    "simplicity_order": list(SIMPLICITY_ORDER),
    "selection_rule": ["1. discard every candidate that is not non-inferior to B2-v1 in both domains",
                       "2. among the rest, find the lowest primary expected escalation",
                       "3. every candidate within 0.005 of it is tied; select the first tied one in the simplicity order",
                       "4. if no candidate is non-inferior, none is promotable: the lowest-escalation candidate is "
                       "still frozen, marked not promotable, so that the regression guard is documented",
                       "C0 is a control, never selected"],
    "fidelity_improvement": "pooled DEV REAL log loss at least 0.01 below B2-v1",
    "regression_guard": {
        "data": ["Stage-4C REAL CLEAN decisions (real_reference export, normalized in memory, frozen clean rule)",
                 "Stage-4C B2-through-R0 decisions as R2.1 replay states"],
        "checks": {**GUARD, "scope": "fidelity checks per level (level_02, level_03); escalation pooled"},
        "descriptive": ["macro F1", "Brier", "JSD", "opportunity rates", "trigger breakdown", "coverage of escalated "
                        "states", "sequence proxies"],
        "timing": "after the candidate manifest is frozen and verified; read once",
        "on_failure": "class C; the selection is never revisited"},
    "decision_classes": {
        "precedence": "A > C > B > D",
        "A": "the selected candidate is non-inferior on DEV, materially reduces the primary expected escalation and "
             "passes the Stage-4C guard",
        "C": "escalation drops but fidelity degrades: a candidate materially reduces the primary expected escalation "
             "but fails DEV non-inferiority, or the selected (non-inferior, materially reducing) candidate fails the "
             "Stage-4C guard",
        "B": "fidelity improves but escalation does not: the selected candidate is non-inferior with a pooled DEV log-"
             "loss improvement >= 0.01 and no material reduction",
        "D": "no meaningful improvement (otherwise)"},
    "stage4c": "regression-only: never read before the freeze; never used for training, selection, hyperparameters, "
               "features, thresholds or the router",
}


def _now() -> str:
    return pr._now()


def _sha(path: Path) -> str:
    return pr._sha(path)


def code_hash() -> dict[str, Any]:
    files = {f: _sha(PROJECT_ROOT / f) for f in CODE_FILES}
    return {"files": files, "hash": content_hash(files)}


# -- the Stage-4C file-open trace ------------------------------------------------------------------------------

class ProtectedTrace:
    """An audit hook: every open / directory listing of a path naming Stage 4C while a phase is active. Python source
    files are code, not data, and are not counted."""

    EVENTS = ("open", "os.listdir", "os.scandir", "glob.glob")

    def __init__(self) -> None:
        self.phase: str | None = None
        self.opened: dict[str, list[str]] = defaultdict(list)
        self.phases: list[str] = []
        self._installed = False

    def install(self) -> "ProtectedTrace":
        if not self._installed:
            sys.addaudithook(self)
            self._installed = True
        return self

    def __call__(self, event: str, args: tuple) -> None:
        if self.phase is None or event not in self.EVENTS or not args:
            return
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            s = os.fsdecode(p)
            if "stage4c" in s.lower() and not s.endswith((".py", ".pyc")):
                self.opened[self.phase].append(s)

    @contextmanager
    def watch(self, phase: str) -> Iterator["ProtectedTrace"]:
        self.install()
        self.phase, prev = phase, self.phase
        self.phases.append(phase)
        self.opened.setdefault(phase, [])
        try:
            yield self
        finally:
            self.phase = prev

    def record(self, phase: str) -> dict[str, Any]:
        return {"phase": phase, "hook_installed": self._installed, "stage4c_paths_opened": sorted(set(self.opened.get(phase, [])))}


TRACE = ProtectedTrace()


def _assert_untouched(phase: str) -> dict[str, Any]:
    rec = TRACE.record(phase)
    if rec["stage4c_paths_opened"]:
        raise WorkshopError(f"STOP: Stage-4C data opened during {phase}: {rec['stage4c_paths_opened'][:3]}")
    return rec


# -- protocol --------------------------------------------------------------------------------------------------

def write_protocol() -> dict[str, Any]:
    pol, split = pr.load_policy(), pr.load_split()
    if CANDIDATE_ROOT.exists():
        raise WorkshopError("candidates exist already; the selection protocol must be frozen before training")
    return pr._freeze(PROTOCOL_PATH, {**PROTOCOL, "promotion_policy_hash": pol["policy_hash"],
                                      "split_hash": split["split_hash"]}, "protocol_hash")


def load_protocol() -> dict[str, Any]:
    body = pr._load_frozen(PROTOCOL_PATH, "protocol_hash")
    if {k: v for k, v in body.items() if k not in ("protocol_hash", "frozen_utc", "promotion_policy_hash", "split_hash")} != \
            json.loads(json.dumps(PROTOCOL)):
        raise WorkshopError("the selection protocol in code differs from the frozen protocol")
    return body


# -- models ----------------------------------------------------------------------------------------------------

def fit_weighted(rows: list[dict], weights: list[float] | None) -> LogisticPolicy:
    """LogisticPolicy.fit with per-row weights (identical code path when weights is None)."""
    model = LogisticPolicy(**CONFIG)
    if weights is None:
        return model.fit(rows)
    keep = [(r, w) for r, w in zip(rows, weights) if r["intended_action"] in model.actions]
    rows, weights = [r for r, _ in keep], [w for _, w in keep]
    model.encoder = model._encoder().fit([r["features"] for r in rows])
    model.classes = sorted({r["intended_action"] for r in rows}, key=model.actions.index)
    X = [model.encoder.transform(r["features"]) for r in rows]
    y = [model.classes.index(r["intended_action"]) for r in rows]
    model.W, model.b, model.info = fit_softmax(X, y, len(model.encoder.columns), len(model.classes), c=model.c,
                                               weights=weights)
    model.n_train = len(rows)
    return model


def _weights(n_phase_a: int, n_hard: int, w: float) -> list[float]:
    raw = [1.0] * n_phase_a + [w] * n_hard
    scale = len(raw) / sum(raw)
    return [v * scale for v in raw]


def b2_v1() -> dict[str, LogisticPolicy]:
    return {k: load_policy("phase-a-v2", k, emulator=EMULATOR, target=TARGET, coverage=False).action_policy
            for k in schema.all_policy_keys()}


def b2_v1_hashes() -> dict[str, str]:
    return {k: load_policy("phase-a-v2", k, emulator=EMULATOR, target=TARGET, coverage=False).metadata["content_hash"]
            for k in schema.all_policy_keys()}


def b1() -> dict[str, Any]:
    return {k: load_policy("phase-a-v2", k, emulator=B1_EMULATOR, target=TARGET, coverage=False).action_policy
            for k in schema.all_policy_keys()}


def _candidate_dir(root: Path, key: str) -> Path:
    model, personality = schema.parse_policy_key(key)
    return root / model / personality


def load_models(root: Path) -> dict[str, LogisticPolicy]:
    out = {}
    for k in schema.all_policy_keys():
        policy, _ = _load(_candidate_dir(root, k), "action", EMULATOR, k, None)
        out[k] = policy
    return out


def _train_rows() -> tuple[list[dict], list[dict], dict[str, Any]]:
    data = pr.load_data_manifest()
    phase_a = ds.load("phase-a-v2", "decisions_clean.jsonl")
    if ds.load_provenance("phase-a-v2")["dataset_hash"] != data["phase_a_clean"]["dataset_hash"]:
        raise WorkshopError("Phase-A CLEAN differs from the materialized data manifest")
    hard = [r for r in pr.load_rows("hard_examples.jsonl") if r.get("status") == "kept" and r["role"] == "train"]
    if any(r["source"] == "stage4c" or "stage4c" in str(r.get("match_dir", "")) for r in hard):
        raise WorkshopError("STOP: a Stage-4C record in the TRAIN hard examples")
    return phase_a, hard, data


def train() -> dict[str, Any]:
    """Train every candidate once (refuses if any exists), check C0 against B2-v1, record everything."""
    with TRACE.watch("train"):
        prot = load_protocol()
        if CANDIDATE_ROOT.exists():
            raise WorkshopError(f"{CANDIDATE_ROOT} exists; candidates are trained once")
        phase_a, hard, data = _train_rows()
        pol, split = pr.load_policy(), pr.load_split()
        started, t0 = _now(), time.monotonic()
        per: dict[str, Any] = {}
        for cid, cand in CANDIDATES.items():
            out_root = CANDIDATE_ROOT / cid
            per[cid] = {}
            for key in schema.all_policy_keys():
                pa = [r for r in phase_a if r["policy_key"] == key]
                hk = [r for r in hard if r["policy_key"] == key] if cand["hard_examples"] else []
                rows = pa + hk
                weights = _weights(len(pa), len(hk), cand["hard_weight"]) if cand["hard_examples"] and cand["hard_weight"] != 1.0 else None
                ts = time.monotonic()
                model = fit_weighted(rows, weights)
                model_name, personality = schema.parse_policy_key(key)
                identity = {
                    "artifact_version": ARTIFACT_VERSION, "policy_key": key,
                    "policy_id": schema.policy_id(model_name, personality, f"stage5a-{cid}"), "model": model_name,
                    "model_id": schema.MODEL_BY_NAME[model_name]["model_id"], "personality": personality,
                    "candidate": cid, "candidate_definition": cand["definition"],
                    "training_dataset": "phase-a-v2" if cid == "C0" else f"stage5a:{cid}",
                    "training_dataset_hash": data["phase_a_clean"]["dataset_hash"] if cid == "C0" else data["data_hash"],
                    "phase_a_dataset_hash": data["phase_a_clean"]["dataset_hash"], "stage5a_data_hash": data["data_hash"],
                    "promotion_policy_hash": pol["policy_hash"], "split_hash": split["split_hash"],
                    "selection_protocol_hash": prot["protocol_hash"], "training_records": len(rows),
                    "training_composition": {"phase_a_clean": len(pa), "hard_examples": len(hk),
                                             "hard_by_source": dict(sorted(Counter(r["source"] for r in hk).items()))},
                    "hard_weight": cand["hard_weight"], "weight_normalization": "mean 1" if weights else None,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION, "created_utc": _now()}
                path = _candidate_dir(out_root, key) / EMULATOR
                _write(path, "action", EMULATOR, model.to_json(), identity, dict(CONFIG))
                (path / "feature_schema.json").write_text(json.dumps(feature_schema(), indent=2) + "\n")
                summary = {"policy_key": key, "candidate": cid, "rows": len(rows), "phase_a_rows": len(pa),
                           "hard_rows": len(hk), "weights": {"phase_a": round(weights[0], 6) if weights else 1.0,
                                                             "hard": round(weights[-1], 6) if weights and hk else (1.0 if hk else None)},
                           "actions": dict(Counter(r["intended_action"] for r in rows).most_common()),
                           "columns": len(model.encoder.columns), "classes": model.classes, "optimizer": model.info,
                           "seconds": round(time.monotonic() - ts, 2)}
                (path / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                per[cid][key] = {**summary, "content_hash": content_hash(model.to_json())}
        c0 = c0_check()
        trace = _assert_untouched("train")
        body = {"version": "stage5a-training-v1", "started_utc": started, "finished_utc": _now(),
                "seconds": round(time.monotonic() - t0, 1), "selection_protocol_hash": prot["protocol_hash"],
                "promotion_policy_hash": pol["policy_hash"], "split_hash": split["split_hash"], "data_hash": data["data_hash"],
                "phase_a_dataset_hash": data["phase_a_clean"]["dataset_hash"], "config": CONFIG,
                "candidates": {cid: {**CANDIDATES[cid], "identities": per[cid],
                                     "rows": sum(v["rows"] for v in per[cid].values()),
                                     "hard_rows": sum(v["hard_rows"] for v in per[cid].values())} for cid in CANDIDATES},
                "c0_reproduction": c0, "code": code_hash(), "stage4c_trace": trace}
        if not c0["pass"]:
            pr._write_json(TRAINING_RECORD, body)
            raise WorkshopError("STOP: C0 does not reproduce B2-v1 (see training_record.json)")
        pr._write_json(TRAINING_RECORD, body)
        return body


def c0_check() -> dict[str, Any]:
    ref, c0 = b2_v1(), load_models(CANDIDATE_ROOT / "C0")
    hashes = b2_v1_hashes()
    rows = ds.load("phase-a-v2", "decisions_clean.jsonl")
    dev = pr.load_rows("dev_real_decisions.jsonl")
    per = {}
    for k in schema.all_policy_keys():
        meta = json.loads((_candidate_dir(CANDIDATE_ROOT / "C0", k) / EMULATOR / "metadata.json").read_text())
        dmax = 0.0
        for r in [x for x in rows if x["policy_key"] == k]:
            legal = pr.legal_action_set(r["observation"])
            a, b = ref[k].predict(r["features"], legal).probs, c0[k].predict(r["features"], legal).probs
            dmax = max(dmax, max(abs(a.get(x, 0.0) - b.get(x, 0.0)) for x in set(a) | set(b)))
        for r in [x for x in dev if x["policy_key"] == k]:
            a, b = ref[k].predict(r["features"], r["legal"]).probs, c0[k].predict(r["features"], r["legal"]).probs
            dmax = max(dmax, max(abs(a.get(x, 0.0) - b.get(x, 0.0)) for x in set(a) | set(b)))
        per[k] = {"b2_v1_content_hash": hashes[k], "c0_content_hash": meta["content_hash"],
                  "identical_content_hash": hashes[k] == meta["content_hash"], "max_abs_prob_diff": dmax}
    ok = all(v["identical_content_hash"] or v["max_abs_prob_diff"] <= 1e-6 for v in per.values())
    return {"pass": ok, "identities": per, "identical_hashes": sum(v["identical_content_hash"] for v in per.values()),
            "max_abs_prob_diff": max(v["max_abs_prob_diff"] for v in per.values()), "rows_checked": len(rows) + len(dev)}


# -- fidelity --------------------------------------------------------------------------------------------------

def _result(policy: Any, row: dict, legal: list[str] | None = None) -> dict[str, Any]:
    pred = policy.predict(row["features"], legal if legal is not None else row["legal"])
    real = row["intended_action"]
    return {"probs": pred.probs, "top": pred.top_action, "real": real, "p_real": pred.probs.get(real, 0.0),
            "masking_changed_top": pred.masking_changed_top, "backoff_level": pred.backoff_level, "features": row["features"]}


def fidelity(models: dict[str, Any], rows: list[dict]) -> dict[str, Any]:
    res = [_result(models[r["policy_key"]], r) for r in rows]
    m = _metrics(res)
    opp = opportunity_rates(res)
    errs = [v["abs_error"] for v in opp.values() if v["n"]]
    m["opportunity_mae"] = round(sum(errs) / len(errs), 4) if errs else None
    m["opportunities"] = opp
    return m


def _short(m: dict[str, Any]) -> dict[str, Any]:
    return {k: m.get(k) for k in ("n", "log_loss", "top1", "macro_f1", "brier", "jsd", "opportunity_mae")}


# -- R2.1 replay -----------------------------------------------------------------------------------------------

def _executable(legal: list[str], unsupported: list[str]) -> list[str]:
    return [a for a in legal if a not in (unsupported or [])] or ["wait"]


def _terms(policy: Any, b1_policy: Any, features: dict, executable: list[str], history: list[str]) -> dict[str, Any]:
    p = policy.predict(features, executable).probs
    q = b1_policy.predict(features, executable).probs
    d = round(rt.tvd(q, p), 6)
    dis = d > rt.TVD_THRESHOLD
    tail = history[-rt.REPETITION_WINDOW:]
    rep_action = tail[-1] if len(tail) == rt.REPETITION_WINDOW and len(set(tail)) == 1 else None
    p_rep = p.get(rep_action, 0.0) if rep_action else 0.0
    return {"tvd": d, "disagreement": dis, "rep_action": rep_action, "p_rep": p_rep,
            "p_esc": 1.0 if dis else p_rep, "p_repeat_last": p.get(history[-1], 0.0) if history else 0.0}


def _aggregate(terms: list[dict], coverage: list[str | None]) -> dict[str, Any]:
    n = len(terms)
    if not n:
        return {"n": 0}
    esc = sum(t["p_esc"] for t in terms)
    both = sum(t["p_rep"] for t in terms if t["disagreement"])
    dis_only = sum(1 - t["p_rep"] for t in terms if t["disagreement"])
    rep_only = sum(t["p_rep"] for t in terms if not t["disagreement"])
    by_cov: dict[str, float] = defaultdict(float)
    n_cov: Counter = Counter()
    for t, c in zip(terms, coverage):
        by_cov[str(c)] += t["p_esc"]
        n_cov[str(c)] += 1
    return {"n": n, "expected_escalation": round(esc / n, 4), "expected_escalations": round(esc, 2),
            "disagreement_rate": round(sum(t["disagreement"] for t in terms) / n, 4),
            "repetition_rate": round(sum(t["p_rep"] for t in terms) / n, 4),
            "both": round(both / n, 4), "disagreement_only": round(dis_only / n, 4), "repetition_only": round(rep_only / n, 4),
            "shares": {"both": round(both / esc, 4) if esc else None, "disagreement_only": round(dis_only / esc, 4) if esc else None,
                       "repetition_only": round(rep_only / esc, 4) if esc else None},
            "coverage_of_escalated": {c: round(v / esc, 4) if esc else None for c, v in sorted(by_cov.items())},
            "escalation_by_coverage": {c: round(by_cov[c] / n_cov[c], 4) for c in sorted(n_cov)},
            "mean_tvd": round(sum(t["tvd"] for t in terms) / n, 4),
            "sequence": {"p_repeat_last_committed": round(sum(t["p_repeat_last"] for t in terms) / n, 4)}}


def replay(models: dict[str, Any], b1_models: dict[str, Any], states: list[dict]) -> tuple[list[dict], list[str]]:
    """Terms per state, states in event order; the history of each (configuration, actor) is the logged one."""
    hist: dict[tuple, list[str]] = defaultdict(list)
    out, cov = [], []
    for s in states:
        h = hist[(s["config_id"], s["actor_id"])]
        t = _terms(models[s["policy_key"]], b1_models[s["policy_key"]], s["features"],
                   _executable(s["legal"], s["unsupported"]), h)
        t["proposed_repeats"] = t["rep_action"] is not None and s.get("proposed") == t["rep_action"]
        out.append(t)
        cov.append(s.get("coverage_label"))
        h.append(s["committed"])
    return out, cov


def replay_check(terms: list[dict], states: list[dict]) -> dict[str, Any]:
    """B2-v1 through the replay vs the signals the live R0 matches logged."""
    dis = sum(1 for t, s in zip(terms, states) if t["disagreement"] != bool(s["logged_signals"]["strong_b1_b2_disagreement"]))
    tvd = sum(1 for t, s in zip(terms, states) if s.get("logged_b1_b2_tvd") is not None and t["tvd"] != s["logged_b1_b2_tvd"])
    rep = sum(1 for t, s in zip(terms, states) if t["proposed_repeats"] != bool(s["logged_signals"]["repetition_loop"]))
    logged = sum(1 for s in states if s["logged_signals"]["strong_b1_b2_disagreement"] or s["logged_signals"]["repetition_loop"])
    return {"states": len(states), "disagreement_mismatches": dis, "tvd_mismatches": tvd, "repetition_mismatches": rep,
            "ok": dis == 0 and tvd == 0 and rep == 0,
            "logged_r21_trigger_rate": round(logged / len(states), 4) if states else None}


def hard_state_terms(models: dict[str, Any], b1_models: dict[str, Any], rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        t = _terms(models[r["policy_key"]], b1_models[r["policy_key"]], r["features"],
                   _executable(r["legal"], r.get("unsupported") or []), r.get("history_tail") or [])
        t["proposed_repeats"] = t["rep_action"] is not None and r.get("b2_v1_proposed") == t["rep_action"]
        out.append(t)
    return out


def hard_check(terms: list[dict], rows: list[dict]) -> dict[str, Any]:
    """B2-v1 at the escalated hard states vs the logged router signals (checks the reconstructed history too)."""
    sig = [((r["trigger"].get("router_signals") or {}).get("strong_b1_b2_disagreement"),
            (r["trigger"].get("router_signals") or {}).get("repetition_loop")) for r in rows]
    dis = sum(1 for t, (d, _) in zip(terms, sig) if t["disagreement"] != bool(d))
    rep = sum(1 for t, (_, p) in zip(terms, sig) if t["proposed_repeats"] != bool(p))
    return {"states": len(rows), "disagreement_mismatches": dis, "repetition_mismatches": rep, "ok": dis == 0 and rep == 0}


def _by_category(rows: list[dict], res: list[dict], terms: list[dict]) -> dict[str, Any]:
    cats = sorted({c for r in rows for c in r["hard_categories"]} | {"all"})
    out = {}
    for c in cats:
        idx = [i for i, r in enumerate(rows) if c == "all" or c in r["hard_categories"]]
        sub = [res[i] for i in idx]
        m = _metrics(sub)
        out[c] = {"n": len(idx), "top1": m.get("top1"), "log_loss": m.get("log_loss"),
                  "expected_escalation": round(sum(terms[i]["p_esc"] for i in idx) / len(idx), 4) if idx else None}
    return out


# -- development evaluation + selection --------------------------------------------------------------------------

def evaluate_development(models_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    b1_models = b1()
    dev_real = pr.load_rows("dev_real_decisions.jsonl")
    replay_rows = pr.load_rows("dev_replay_r0.jsonl")
    dev_hard = [r for r in pr.load_rows("hard_examples.jsonl") if r.get("status") == "kept" and r["role"] == "development"]
    train_hard = [r for r in pr.load_rows("hard_examples.jsonl") if r.get("features") and r["role"] in ("train", "development")
                  and r.get("observation_key")]
    out: dict[str, Any] = {"rows": {"dev_real": dict(Counter(r["source"] for r in dev_real)),
                                    "dev_replay": dict(Counter(r["source"] for r in replay_rows)), "dev_hard": len(dev_hard)}}
    checks = {}
    for mid, models in models_by_id.items():
        fid = {dom: fidelity(models, [r for r in dev_real if r["source"] == src]) for dom, src in DOMAINS.items()}
        fid["pooled"] = fidelity(models, dev_real)
        terms, cov = replay(models, b1_models, replay_rows)
        esc = {"pooled": _aggregate(terms, cov)}
        for dom, src in REPLAY_DOMAINS.items():
            idx = [i for i, s in enumerate(replay_rows) if s["source"] == src]
            esc[dom] = _aggregate([terms[i] for i in idx], [cov[i] for i in idx])
        h_terms = hard_state_terms(models, b1_models, dev_hard)
        h_res = [_result(models[r["policy_key"]], r) for r in dev_hard]
        real_dis = [round(rt.tvd(b1_models[r["policy_key"]].predict(r["features"], _executable(r["legal"], r["unsupported"])).probs,
                                 models[r["policy_key"]].predict(r["features"], _executable(r["legal"], r["unsupported"])).probs), 6)
                    > rt.TVD_THRESHOLD for r in dev_real]
        out[mid] = {"fidelity": {k: {**_short(v), "opportunities": v["opportunities"]} for k, v in fid.items()},
                    "escalation": esc,
                    "hard_states": {"fidelity": _short(_metrics(h_res)), "by_category": _by_category(dev_hard, h_res, h_terms),
                                    "by_trigger": {cls: _aggregate([h_terms[i] for i, r in enumerate(dev_hard) if r["trigger"]["reason_class"] == cls],
                                                                   [r["trigger"]["coverage_label"] for r in dev_hard if r["trigger"]["reason_class"] == cls])
                                                   for cls in sorted({r["trigger"]["reason_class"] for r in dev_hard})},
                                    "expected_escalation": round(sum(t["p_esc"] for t in h_terms) / len(h_terms), 4) if h_terms else None},
                    "dev_real_disagreement_rate": round(sum(real_dis) / len(real_dis), 4) if real_dis else None}
        if mid == "B2-v1":
            checks["r0_replay"] = replay_check(terms, replay_rows)
            all_terms = hard_state_terms(models, b1_models, train_hard)
            checks["hard_states"] = hard_check(all_terms, train_hard)
    out["replay_checks"] = checks
    if not checks["r0_replay"]["ok"]:
        raise WorkshopError(f"STOP: the analytic replay does not reproduce the logged R2.1 signals: {checks['r0_replay']}")
    return out


def _ni(c: dict[str, Any], b: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    checks, ok = {}, True
    for dom in DOMAINS:
        for metric, (direction, margin) in NON_INFERIORITY.items():
            cv, bv = c["fidelity"][dom][metric], b["fidelity"][dom][metric]
            worse = round((cv - bv) if direction == "lower" else (bv - cv), 4)
            passed = worse <= margin
            checks[f"{dom}.{metric}"] = {"candidate": cv, "b2_v1": bv, "worse_by": worse, "margin": margin, "pass": passed}
            ok = ok and passed
    return ok, checks


def select(dev: dict[str, Any]) -> dict[str, Any]:
    base = dev["B2-v1"]
    b_esc = base["escalation"]["pooled"]["expected_escalation"]
    rows = {}
    for cid in SIMPLICITY_ORDER:
        c = dev[cid]
        ok, checks = _ni(c, base)
        e = c["escalation"]["pooled"]["expected_escalation"]
        red = round(b_esc - e, 4)
        rel = round(red / b_esc, 4) if b_esc else None
        rows[cid] = {"non_inferior": ok, "checks": checks, "failed_checks": [k for k, v in checks.items() if not v["pass"]],
                     "expected_escalation": e, "absolute_reduction": red, "relative_reduction": rel,
                     "material_reduction": red >= MATERIAL["absolute"] and rel is not None and rel >= MATERIAL["relative"],
                     "pooled_log_loss_improvement": round(base["fidelity"]["pooled"]["log_loss"] - c["fidelity"]["pooled"]["log_loss"], 4)}
    eligible = [cid for cid in SIMPLICITY_ORDER if rows[cid]["non_inferior"]]
    pool = eligible or list(SIMPLICITY_ORDER)
    best = min(rows[cid]["expected_escalation"] for cid in pool)
    tied = [cid for cid in SIMPLICITY_ORDER if cid in pool and rows[cid]["expected_escalation"] - best <= TIE_MARGIN + 1e-12]
    selected = tied[0]
    return {"b2_v1_expected_escalation": b_esc, "candidates": rows, "non_inferior": eligible, "lowest_expected_escalation": best,
            "tied": tied, "selected": selected, "promotable": bool(eligible),
            "selected_material_reduction": rows[selected]["material_reduction"],
            "rule": PROTOCOL["selection_rule"]}


def frontier(dev: dict[str, Any]) -> list[dict[str, Any]]:
    pts = []
    for mid in ("B2-v1", *CANDIDATES):
        f, e = dev[mid]["fidelity"], dev[mid]["escalation"]["pooled"]
        pts.append({"model": mid, "expected_escalation": e["expected_escalation"], "log_loss": f["pooled"]["log_loss"],
                    "top1": f["pooled"]["top1"], "jsd": f["pooled"]["jsd"],
                    "log_loss_level_01": f["level_01"]["log_loss"], "log_loss_shifted": f["shifted"]["log_loss"]})
    for p in pts:  # Pareto: no other point at most as escalating with lower-or-equal log loss, one strictly better
        p["pareto"] = not any(q is not p and q["expected_escalation"] <= p["expected_escalation"] and q["log_loss"] <= p["log_loss"]
                              and (q["expected_escalation"] < p["expected_escalation"] or q["log_loss"] < p["log_loss"]) for q in pts)
    return pts


def develop_and_select() -> dict[str, Any]:
    """DEVELOPMENT evaluation of B2-v1 and every candidate, the selection, and the freeze; once."""
    with TRACE.watch("select"):
        prot = load_protocol()
        rec = json.loads(TRAINING_RECORD.read_text())
        if not rec["c0_reproduction"]["pass"]:
            raise WorkshopError("STOP: C0 did not reproduce B2-v1")
        if DEVELOPMENT_PATH.is_file() or FREEZE_DIR.exists():
            raise WorkshopError("the development selection and the freeze are written once")
        for cid in CANDIDATES:
            for k, v in rec["candidates"][cid]["identities"].items():
                meta = json.loads((_candidate_dir(CANDIDATE_ROOT / cid, k) / EMULATOR / "metadata.json").read_text())
                if meta["content_hash"] != v["content_hash"]:
                    raise WorkshopError(f"{cid} {k}: the candidate artifact changed after training")
        models = {"B2-v1": b2_v1(), **{cid: load_models(CANDIDATE_ROOT / cid) for cid in CANDIDATES}}
        t0 = time.monotonic()
        dev = evaluate_development(models)
        sel = select(dev)
        dev_trace = _assert_untouched("select")
        body = {"version": "stage5a-development-v1", "selection_protocol_hash": prot["protocol_hash"],
                "training_record_sha256": _sha(TRAINING_RECORD), "data_hash": rec["data_hash"], "split_hash": rec["split_hash"],
                "results": dev, "selection": sel, "frontier": frontier(dev), "seconds": round(time.monotonic() - t0, 1),
                "stage4c_trace": dev_trace, "evaluated_utc": _now()}
        pr._write_json(DEVELOPMENT_PATH, body)
        manifest = freeze(sel, rec, prot)
        _assert_untouched("select")
        return {**body, "freeze": manifest}


# -- freeze ----------------------------------------------------------------------------------------------------

def freeze(sel: dict[str, Any], rec: dict[str, Any], prot: dict[str, Any]) -> dict[str, Any]:
    cid = sel["selected"]
    if FREEZE_MANIFEST.is_file():
        raise WorkshopError(f"{FREEZE_MANIFEST} exists; the candidate is frozen once")
    src = CANDIDATE_ROOT / cid
    shutil.copytree(src, FREEZE_DIR)
    pol, split, data = pr.load_policy(), pr.load_split(), pr.load_data_manifest()
    dev = json.loads(DEVELOPMENT_PATH.read_text())
    candidate = {
        "candidate_id": "b2-v2-candidate", "selected": cid, "definition": CANDIDATES[cid]["definition"],
        "promotable": sel["promotable"], "status": "frozen B2-v2 candidate (offline; not confirmed)" if sel["promotable"]
        else "frozen for the documented regression guard only: no candidate was non-inferior (not promotable)",
        "model_family": EMULATOR, "hyperparameters": {**CONFIG, "epsilon": 1e-3, "optimizer": "L-BFGS max_iter 200 tol 1e-6"},
        "weights": {"hard_weight": CANDIDATES[cid]["hard_weight"], "normalization": PROTOCOL["weighting"],
                    "per_identity": {k: v["weights"] for k, v in rec["candidates"][cid]["identities"].items()}},
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_schema": feature_schema(),
        "target_policy": "target-v2.1 (frozen B2-v1 artifact, not retrained)", "router": f"R2.1 {R21_HASH} (unchanged)",
        "training_data": {"phase_a_dataset_hash": data["phase_a_clean"]["dataset_hash"], "stage5a_data_hash": data["data_hash"],
                          "hard_examples_sha256": data["files"]["hard_examples.jsonl"]["sha256"],
                          "rows": rec["candidates"][cid]["rows"], "hard_rows": rec["candidates"][cid]["hard_rows"]},
        "development_split_hash": split["split_hash"], "promotion_policy_hash": pol["policy_hash"],
        "selection_protocol_hash": prot["protocol_hash"], "training_code": rec["code"],
        "training_timestamp": rec["finished_utc"], "training_record_sha256": _sha(TRAINING_RECORD),
        "selection_record": {"path": str(DEVELOPMENT_PATH.relative_to(PROJECT_ROOT)), "sha256": _sha(DEVELOPMENT_PATH),
                             **{k: dev["selection"][k] for k in ("selected", "non_inferior", "tied", "promotable",
                                                                 "b2_v1_expected_escalation", "lowest_expected_escalation")},
                             "selected_row": dev["selection"]["candidates"][cid]},
        "identities": {k: v["content_hash"] for k, v in rec["candidates"][cid]["identities"].items()},
        "stage4c": "not read before this freeze (trace: no Stage-4C path opened in train / select)",
    }
    pr._write_json(FREEZE_DIR / "candidate.json", candidate)
    files = sorted(p for p in FREEZE_DIR.rglob("*") if p.is_file() and p.name != "manifest.json")
    body = {"version": "stage5a-b2-v2-candidate-manifest-v1", "candidate": cid,
            "files": {str(p.relative_to(FREEZE_DIR)): _sha(p) for p in files}, "selection_protocol_hash": prot["protocol_hash"],
            "promotion_policy_hash": pol["policy_hash"], "split_hash": split["split_hash"], "data_hash": data["data_hash"],
            "training_code_hash": rec["code"]["hash"]}
    body = {**body, "manifest_hash": content_hash(body), "frozen_utc": _now()}
    pr._write_json(FREEZE_MANIFEST, body)
    return body


def verify_freeze() -> dict[str, Any]:
    body = json.loads(FREEZE_MANIFEST.read_text())
    problems = []
    if content_hash({k: v for k, v in body.items() if k not in ("manifest_hash", "frozen_utc")}) != body["manifest_hash"]:
        problems.append("manifest_hash")
    for rel, h in body["files"].items():
        p = FREEZE_DIR / rel
        if not p.is_file() or _sha(p) != h:
            problems.append(rel)
    extra = sorted(str(p.relative_to(FREEZE_DIR)) for p in FREEZE_DIR.rglob("*") if p.is_file()
                   and p.name != "manifest.json" and str(p.relative_to(FREEZE_DIR)) not in body["files"])
    problems += [f"unlisted {x}" for x in extra]
    return {"ok": not problems, "problems": problems, "manifest_hash": body["manifest_hash"], "candidate": body["candidate"],
            "frozen_utc": body["frozen_utc"]}


# -- Stage-4C regression-only guard ------------------------------------------------------------------------------

def stage4c_real() -> tuple[list[dict], dict[str, Any]]:
    from workshop.policy_emulator.features import PolicyFeatureExtractorV1
    cfg = {"full_match": {"experiment_id": "stage4c-confirmatory:real_reference", "root": STAGE4C_REAL_EXPORT,
                          "decisions": "decisions.jsonl", "manifest": "manifest.json"}}
    recs, sources = ds.normalize_full_match(cfg, "stage5a-regression-stage4c-real", PolicyFeatureExtractorV1(), PROJECT_ROOT)
    rows = [pr._real_row(r, "stage4c_real", pr._level(r["match_id"], None)) for r in recs if r["clean"]]
    return rows, {"path": f"{STAGE4C_REAL_EXPORT}/decisions.jsonl", "sha256": sources["decisions"]["sha256"],
                  "records": len(recs), "clean": len(rows)}


def regression() -> dict[str, Any]:
    """Once, after the freeze verifies: B2-v1 vs the frozen candidate on Stage-4C history; then the decision class."""
    prot = load_protocol()
    v = verify_freeze()
    if not v["ok"]:
        raise WorkshopError(f"STOP: the candidate manifest does not verify: {v['problems']}")
    if REGRESSION_PATH.is_file():
        raise WorkshopError(f"{REGRESSION_PATH} exists; the regression guard is read once")
    dev = json.loads(DEVELOPMENT_PATH.read_text())
    sel = dev["selection"]
    with TRACE.watch("regression"):
        t0 = time.monotonic()
        models = {"B2-v1": b2_v1(), "B2-v2-candidate": load_models(FREEZE_DIR)}
        b1_models = b1()
        real, real_stats = stage4c_real()
        states, st = pr.replay_states("stage4c", "regression-only", allow_protected=True)
        levels = sorted({r["level"] for r in real})
        res: dict[str, Any] = {}
        for mid, m in models.items():
            fid = {lv: fidelity(m, [r for r in real if r["level"] == lv]) for lv in levels}
            fid["pooled"] = fidelity(m, real)
            terms, cov = replay(m, b1_models, states)
            esc = {"pooled": _aggregate(terms, cov)}
            for lv in sorted({s["level"] for s in states}):
                idx = [i for i, s in enumerate(states) if s["level"] == lv]
                esc[lv] = _aggregate([terms[i] for i in idx], [cov[i] for i in idx])
            mean = defaultdict(float)
            for r in real:
                for a, p in m[r["policy_key"]].predict(r["features"], r["legal"]).probs.items():
                    mean[a] += p / len(real)
            res[mid] = {"fidelity": {k: {**_short(x), "opportunities": x["opportunities"]} for k, x in fid.items()},
                        "escalation": esc, "mean_action_distribution": {a: round(p, 4) for a, p in sorted(mean.items())}}
            if mid == "B2-v1":
                res["replay_check"] = replay_check(terms, states)
        trace = TRACE.record("regression")
    c, b = res["B2-v2-candidate"], res["B2-v1"]
    guard = {}
    for lv in levels:
        cf, bf = c["fidelity"][lv], b["fidelity"][lv]
        guard[f"{lv}.log_loss"] = {"candidate": cf["log_loss"], "b2_v1": bf["log_loss"], "change": round(cf["log_loss"] - bf["log_loss"], 4),
                                   "limit": GUARD["log_loss_max_increase"], "pass": round(cf["log_loss"] - bf["log_loss"], 4) <= GUARD["log_loss_max_increase"]}
        guard[f"{lv}.top1"] = {"candidate": cf["top1"], "b2_v1": bf["top1"], "change": round(cf["top1"] - bf["top1"], 4),
                               "limit": -GUARD["top1_max_decrease"], "pass": round(bf["top1"] - cf["top1"], 4) <= GUARD["top1_max_decrease"]}
        guard[f"{lv}.opportunity_mae"] = {"candidate": cf["opportunity_mae"], "b2_v1": bf["opportunity_mae"],
                                          "change": round(cf["opportunity_mae"] - bf["opportunity_mae"], 4),
                                          "limit": GUARD["opportunity_mae_max_increase"],
                                          "pass": round(cf["opportunity_mae"] - bf["opportunity_mae"], 4) <= GUARD["opportunity_mae_max_increase"]}
    ce, be = c["escalation"]["pooled"]["expected_escalation"], b["escalation"]["pooled"]["expected_escalation"]
    guard["pooled.expected_escalation"] = {"candidate": ce, "b2_v1": be, "change": round(ce - be, 4), "limit": 0.0, "pass": ce <= be}
    passed = all(g["pass"] for g in guard.values())
    decision = decide(sel, passed)
    body = {"version": "stage5a-regression-v1", "label": "regression-only - not confirmatory - not used for candidate selection",
            "selection_protocol_hash": prot["protocol_hash"], "candidate_manifest_hash": v["manifest_hash"],
            "candidate": v["candidate"], "freeze_verified": v, "freeze_frozen_utc": v["frozen_utc"],
            "real": real_stats, "replay_states": st, "results": res, "guard": guard, "guard_pass": passed,
            "relative_escalation_reduction": round((be - ce) / be, 4) if be else None,
            "decision": decision, "stage4c_trace": trace, "seconds": round(time.monotonic() - t0, 1), "evaluated_utc": _now()}
    pr._write_json(REGRESSION_PATH, body)
    return body


def decide(sel: dict[str, Any], guard_pass: bool) -> dict[str, Any]:
    rows, cid = sel["candidates"], sel["selected"]
    s = rows[cid]
    reducing_fail_ni = [k for k, r in rows.items() if r["material_reduction"] and not r["non_inferior"]]
    if sel["promotable"] and s["material_reduction"] and guard_pass:
        code, why = "A", f"{cid} is non-inferior on DEV, reduces the expected R2.1 escalation materially and passes the Stage-4C guard"
    elif sel["promotable"] and s["material_reduction"] and not guard_pass:
        code, why = "C", f"{cid} reduces escalation on DEV but fails the Stage-4C regression guard"
    elif reducing_fail_ni:
        code, why = "C", f"material escalation reductions ({', '.join(reducing_fail_ni)}) come only with DEV non-inferiority failures"
    elif sel["promotable"] and s["pooled_log_loss_improvement"] >= FIDELITY_IMPROVEMENT:
        code, why = "B", f"{cid} improves DEV fidelity (pooled log loss -{s['pooled_log_loss_improvement']}) without a material escalation reduction"
    else:
        code, why = "D", "no candidate is both non-inferior and materially less escalating, and none improves fidelity"
    labels = {"A": "viable B2-v2 candidate", "B": "fidelity improves but escalation does not",
              "C": "escalation drops but fidelity degrades", "D": "no meaningful improvement"}
    recs = {"A": "proceed to Stage 5B - call-free closed-loop surrogate selection",
            "B": "analyze router interaction before closed-loop promotion",
            "C": "do not promote the candidate", "D": "stop the active-learning branch and reconsider the data / model strategy"}
    return {"code": code, "label": labels[code], "reason": why, "recommendation": recs[code], "guard_pass": guard_pass,
            "selected": cid, "promotable": sel["promotable"], "precedence": PROTOCOL["decision_classes"]["precedence"]}


__all__ = ["PROTOCOL", "CANDIDATES", "write_protocol", "load_protocol", "train", "c0_check", "develop_and_select",
           "verify_freeze", "regression", "decide", "select", "fidelity", "replay", "TRACE", "fit_weighted", "load_models",
           "b2_v1", "b1", "FREEZE_DIR", "PROTOCOL_PATH", "DEVELOPMENT_PATH", "REGRESSION_PATH", "TRAINING_RECORD"]
