"""Stage 5A - training-data promotion (Archive/STAGE5A_DATA_PROMOTION_AND_OFFLINE_RETRAINING.md sections 3-10, 16).

Freeze order (each artifact hashed and never rewritten):
    workshop/policies/stage5/stage5a_promotion_policy.json   the promotion rules (before any data is materialized)
    workshop/policies/stage5/stage5a_split.json              TRAIN / DEVELOPMENT / REGRESSION-ONLY by whole configuration
    workshop/policy_data/stage5a/                            the materialized hard examples, DEV sets and replay states
    workshop/policy_data/stage5_data_promotion_registry.json every historical source, its role and its eligibility

Offline and read-only towards history: every source is read from its frozen file or telemetry; nothing historical is
rewritten. Stage 4C is never opened here: its registry rows are taken from the frozen Stage-4 release registry.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from workshop.policy_emulator import dataset as ds
from workshop.policy_emulator import schema
from workshop.policy_emulator.artifacts import content_hash
from workshop.policy_emulator.policies import legal_action_set
from workshop.runners.common import PROJECT_ROOT, WorkshopError

STAGE5_DIR = PROJECT_ROOT / "workshop/policies/stage5"
POLICY_PATH = STAGE5_DIR / "stage5a_promotion_policy.json"
SPLIT_PATH = STAGE5_DIR / "stage5a_split.json"
DATA_DIR = PROJECT_ROOT / "workshop/policy_data/stage5a"
DATA_MANIFEST = DATA_DIR / "manifest.json"
REGISTRY_PATH = PROJECT_ROOT / "workshop/policy_data/stage5_data_promotion_registry.json"
PHASE_A = PROJECT_ROOT / "workshop/policy_data/phase-a-v2/decisions_clean.jsonl"
M4D_REAL = PROJECT_ROOT / "workshop/policy_data/hybrid-r21-confirmatory-real-v1/decisions_clean.jsonl"
STAGE4B_REAL_EXPORT = "workshop_runs/stage4b-provider-pilot/real_reference"
ROLES = ("train", "development", "regression-only", "excluded", "invalid", "engineering-only")
PROTECTED_MARKERS = ("stage4c",)  # any path containing these is Stage-4C data (regression-only)

# Escalation (hard-example) sources: hybrid matches whose LLM decisions were made at router-selected states.
ESCALATION_SOURCES = {
    "m4b": {"stage": "Stage 3 M4B (Hybrid R2 provider-backed pilot)", "level": "level_01", "router": "R2",
            "match_dirs": "workshop_runs/stage3-hybrid-pilot-r2/matches/*",
            "frozen_dataset": "workshop/policy_data/hybrid-escalation-pilot-v1/escalations.jsonl"},
    "m4c": {"stage": "Stage 3 M4C (Hybrid R2.1 held-out validation)", "level": "level_01", "router": "R2.1",
            "match_dirs": "workshop_runs/stage3-hybrid-router-v2-heldout/matches/*",
            "frozen_dataset": "workshop/policy_data/hybrid-router-v2-validation/escalations.jsonl"},
    "m4d": {"stage": "Stage 3 M4D (Hybrid R2.1 confirmatory, valid epoch)", "level": "level_01", "router": "R2.1",
            "match_dirs": "workshop_runs/stage3-hybrid-r21-confirmatory/matches/hybrid/*",
            "frozen_dataset": "workshop/policy_data/hybrid-r21-confirmatory-escalations-v1/escalations.jsonl"},
    "stage4b": {"stage": "Stage 4B (provider-backed pilot, Levels 2 / 3)", "level": None, "router": "R2.1",
                "match_dirs": "workshop_runs/stage4b-provider-pilot/matches/hybrid/*", "frozen_dataset": None},
}
R0_SOURCES = {
    "m4d": {"match_dirs": "workshop_runs/stage3-hybrid-r21-confirmatory/r0_control/*", "level": "level_01"},
    "stage4b": {"match_dirs": "workshop_runs/stage4b-provider-pilot/matches/r0/*", "level": None},
}

POLICY: dict[str, Any] = {
    "version": "stage5a-promotion-policy-v1",
    "spec": "Archive/STAGE5A_DATA_PROMOTION_AND_OFFLINE_RETRAINING.md",
    "eligible_stages": ["Stage 3 Phase A (CLEAN, the B2-v1 training set)", "Stage 3 M4B / M4C / M4D provider-backed "
                        "Hybrid escalations (valid epoch only)", "Stage 4B provider-backed pilot escalations"],
    "never_eligible": ["Stage 4C (any arm, any derivative): protected, regression-only",
                       "M4D failed epoch 1 (invalid infrastructure)", "every engineering-check-only session",
                       "every invalid / superseded artifact", "sources without a REAL LLM label (B1 / B2 / R0 / surrogate "
                       "decisions, Stage-4A call-free data)"],
    "eligible_record_types": {
        "phase_a_clean": "canonical full-match records of decisions_clean.jsonl, unchanged (the B2-v1 training rows)",
        "hard_example": "a hybrid_decision event with decision_source = llm and llm_outcome = committed, joined on "
                        "llm_decision_id to its llm_decision event (observation, legal actions with targets, REAL "
                        "validated action) and to its barrier commit"},
    "required_fields": ["features (policy-features-v1, as the live controller computed them)", "observation (full obs-v1)",
                        "legal_actions with targets", "validated_action (REAL action and target)", "policy identity "
                        "(model:personality)", "seed", "configuration", "level", "actor / side", "decision index",
                        "router reasons", "coverage label", "B1 and B2-v1 distributions"],
    "label": {"source": "llm_decision.validated_action (the REAL LLM's validated action and target)",
              "precedence": "validated_action; the executed action only confirms the commit; a B2 counterfactual is "
                            "never a label"},
    "invalid_record_handling": "the frozen Stage-3 clean-record rule (dataset.full_match_exclusions) applied to the "
                               "joined llm_decision and commit: hard (telemetry_incomplete, fallback, "
                               "action_not_in_legal_set) and soft (repair_after_provider_error, "
                               "repair_after_invalid_output) exclusions both remove the record from promotion",
    "fallback_handling": "excluded (a fallback carries no REAL label)",
    "repair_handling": "excluded (the frozen clean rule: a repaired decision is not CLEAN)",
    "discarded_escalations": "excluded (llm_outcome != committed: the decision never acted)",
    "target_normalization": "none: the action label is the REAL action; the target is kept for provenance only "
                            "(target-v2.1 stays frozen and is not retrained in Stage 5A)",
    "identity_fields": ["policy_key = model:personality (from the hybrid event's policy)", "model_id", "personality",
                        "level", "seed", "config_id", "actor_id", "decision_index", "llm_decision_id"],
    "features": "hybrid_decision.features (what the live surrogate saw, with the hybrid's committed-action history); "
                "checked against PolicyFeatureExtractorV1.extract on the joined observation with that history",
    "dedup_key": "sha256 of the canonical JSON of {policy_key, normalized observation (dataset.project_observation), "
                 "sorted legal (action, target) pairs}",
    "dedup_rule": ["Phase-A CLEAN rows are never changed (C0 must reproduce B2-v1)",
                   "a hard example whose key equals a Phase-A key with the same label: dropped as already represented",
                   "a hard example whose key equals a Phase-A key with another label: conflict, hard example excluded",
                   "hard examples sharing a key with the same label: the first in source order is kept",
                   "hard examples sharing a key with different labels: conflict, every one excluded; label counts and "
                   "entropy recorded"],
    "leakage_rule": "a DEVELOPMENT record whose key equals any TRAIN key is excluded from development evaluation",
    "hard_categories": {
        "B1/B2 strong disagreement": "router reason strong_b1_b2_disagreement",
        "repetition trigger": "router reason repetition_loop",
        "both triggers": "both R2.1 reasons on the same decision",
        "R2-only trigger": "an M4B (router R2) reason that R2.1 does not use (e.g. low_ood_critical)",
        "LOW coverage": "coverage-v1 label LOW", "OOD coverage": "coverage-v1 label OOD",
        "rare legal-action pattern": "a coverage-v1 reason naming an unseen / rare legal-action set",
        "mission-critical state": "router critical_state (time pressure not relaxed, or low health)",
        "invalid B2 target": "the B2-v1 counterfactual target is not a legal target of its action in the observation",
    },
    "future_information": "none: every category is read from the frozen telemetry of the decision itself",
}

SPLIT: dict[str, Any] = {
    "version": "stage5a-split-v1",
    "unit": "whole configuration (seed x personality x orientation [x level]); a seed's configurations never straddle "
            "two roles",
    "train": {"phase_a": [7101, 7202, 7303, 7422, 7435, 7457], "m4b": [7101, 7303], "m4c": [7202, 7422],
              "m4d": [8001, 8002], "stage4b": [9007]},
    "development": {"m4d": [8003, 8004], "stage4b": [9008]},
    "regression_only": {"stage4c": [9009, 9010, 9011, 9012]},
    "rationale": ["M4B / M4C replay Phase-A configurations, so they can only be TRAIN",
                  "M4D (Level 1) and Stage 4B (Levels 2 / 3) are split by seed so that both TRAIN and DEVELOPMENT "
                  "contain Level-1 and shifted-level hard states",
                  "Stage 4C stays regression-only and is never read before the candidate is frozen"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mkdir(path: Path) -> None:
    if not path.is_dir():  # an mkdir of an existing shared directory would count as a write outside the output
        path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, body: Any) -> None:
    _mkdir(path.parent)
    path.write_text(json.dumps(body, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    _mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str, separators=(",", ":")) + "\n")


def _freeze(path: Path, body: dict[str, Any], key: str) -> dict[str, Any]:
    if path.is_file():
        raise WorkshopError(f"{path} is frozen; it is never rewritten")
    body = {**body, key: content_hash(body), "frozen_utc": _now()}
    _write_json(path, body)
    return body


def _load_frozen(path: Path, key: str) -> dict[str, Any]:
    body = json.loads(path.read_text())
    if content_hash({k: v for k, v in body.items() if k not in (key, "frozen_utc")}) != body[key]:
        raise WorkshopError(f"{path}: content hash mismatch")
    return body


def write_policy() -> dict[str, Any]:
    return _freeze(POLICY_PATH, POLICY, "policy_hash")


def load_policy() -> dict[str, Any]:
    body = _load_frozen(POLICY_PATH, "policy_hash")
    if {k: v for k, v in body.items() if k not in ("policy_hash", "frozen_utc")} != POLICY:
        raise WorkshopError("the promotion policy in code differs from the frozen policy")
    return body


def write_split() -> dict[str, Any]:
    pol = load_policy()
    return _freeze(SPLIT_PATH, {**SPLIT, "promotion_policy_hash": pol["policy_hash"]}, "split_hash")


def load_split() -> dict[str, Any]:
    return _load_frozen(SPLIT_PATH, "split_hash")


def role_of(source: str, seed: int) -> str:
    if seed in SPLIT["train"].get(source, []):
        return "train"
    if seed in SPLIT["development"].get(source, []):
        return "development"
    return "excluded"


# -- keys --------------------------------------------------------------------------------------------------

def observation_key(policy_key: str, observation: dict[str, Any], legal_pairs: Iterable[tuple]) -> str:
    body = {"policy_key": policy_key, "observation": ds.project_observation(observation),
            "legal": sorted([list(p) for p in legal_pairs], key=lambda p: (str(p[0]), str(p[1])))}
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _pairs(legal_actions: list[dict]) -> list[tuple]:
    return [(a.get("action"), a.get("target_id")) for a in legal_actions or []]


def _level(config_id: str, default: str | None) -> str:
    if default:
        return default
    for lv in ("level_02", "level_03"):
        if config_id.startswith(lv) or f"-{lv}-" in config_id:
            return lv
    raise WorkshopError(f"no level in {config_id}")


def _seed(config_id: str) -> int:
    for part in config_id.split("-"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    raise WorkshopError(f"no seed in {config_id}")


def _guard_protected(path: Path) -> None:
    if any(m in str(path) for m in PROTECTED_MARKERS):
        raise WorkshopError(f"STOP: {path} is Stage-4C data (regression-only); it is not opened before the freeze")


# -- hard examples -----------------------------------------------------------------------------------------

def _categories(h: dict[str, Any], obs: dict[str, Any]) -> list[str]:
    reasons = set(h.get("router_reasons") or [])
    out = []
    if "strong_b1_b2_disagreement" in reasons:
        out.append("B1/B2 strong disagreement")
    if "repetition_loop" in reasons:
        out.append("repetition trigger")
    if {"strong_b1_b2_disagreement", "repetition_loop"} <= reasons:
        out.append("both triggers")
    if reasons - {"strong_b1_b2_disagreement", "repetition_loop"}:
        out.append("R2-only trigger")
    if h.get("coverage_label") == "LOW":
        out.append("LOW coverage")
    if h.get("coverage_label") == "OOD":
        out.append("OOD coverage")
    if any(str(r).startswith("legal-action set") for r in h.get("coverage_reasons") or []):
        out.append("rare legal-action pattern")
    if h.get("critical_state"):
        out.append("mission-critical state")
    cf = h.get("b2_counterfactual") or {}
    targets = schema.legal_targets(obs).get(cf.get("action")) if cf.get("action") else None
    if cf.get("action") and targets is not None and cf.get("target") not in targets:
        out.append("invalid B2 target")
    return out


def _reason_class(reasons: list[str]) -> str:
    r = set(reasons)
    if {"strong_b1_b2_disagreement", "repetition_loop"} <= r:
        return "both"
    if "strong_b1_b2_disagreement" in r and len(r) == 1:
        return "disagreement only"
    if "repetition_loop" in r and len(r) == 1:
        return "repetition only"
    return "other (R2 signal)" if r else "none"


def extract_escalations(source: str) -> tuple[list[dict], dict[str, Any]]:
    """Every committed LLM escalation of one source, joined to its llm_decision and commit, with the frozen clean rule
    applied and the full trigger provenance kept. Non-committed escalations are counted, never returned."""
    from workshop.policy_emulator.features import PolicyFeatureExtractorV1
    cfg = ESCALATION_SOURCES[source]
    ex = PolicyFeatureExtractorV1()
    frozen_ids = None
    if cfg["frozen_dataset"]:
        frozen_ids = {r["decision_id"] for r in _jsonl(PROJECT_ROOT / cfg["frozen_dataset"])}
    out, stats = [], Counter()
    dirs = sorted(p for p in PROJECT_ROOT.glob(cfg["match_dirs"]) if (p / "telemetry_match01.jsonl").is_file())
    for d in dirs:
        _guard_protected(d)
        config_id = d.name
        events = _jsonl(d / "telemetry_match01.jsonl")
        llm = {e["decision_id"]: e for e in events if e.get("event_type") == "llm_decision"}
        commits = {e["decision_id"]: e for e in events if e.get("event_type") in ("barrier_commit", "live_async_commit")
                   and e.get("decision_id")}
        history: dict[str, list[str]] = defaultdict(list)
        for h in (e for e in events if e.get("event_type") == "hybrid_decision"):
            actor = h["actor_id"]
            if h.get("decision_source") == "llm":
                stats["llm_source_events"] += 1
                if h.get("llm_outcome") != "committed":
                    stats["not_committed"] += 1
                    continue
                raw = llm.get(h.get("llm_decision_id"))
                commit = commits.get(h.get("llm_decision_id"))
                hard, soft = ds.full_match_exclusions(raw, commit)
                obs = (raw or {}).get("observation") or {}
                validated = (raw or {}).get("validated_action") or {}
                recomputed = ex.extract(obs, list(reversed(history[actor]))[:3]) if obs else None
                policy_key = h["policy"]
                seed = _seed(config_id)
                rec = {
                    "source": source, "source_stage": cfg["stage"], "router": cfg["router"], "config_id": config_id,
                    "match_dir": str(d.relative_to(PROJECT_ROOT)), "level": _level(config_id, cfg["level"]),
                    "seed": seed, "role": role_of(source, seed), "policy_key": policy_key,
                    "model": policy_key.split(":")[0], "personality": policy_key.split(":")[1],
                    "model_id": (raw or {}).get("model"), "actor_id": actor, "side": actor.split("_")[0],
                    "orientation": config_id.rsplit("-", 1)[-1], "decision_index": h.get("decision_index"),
                    "hybrid_decision_id": h.get("decision_id"), "llm_decision_id": h.get("llm_decision_id"),
                    "trigger": {"reasons": h.get("router_reasons") or [], "reason_class": _reason_class(h.get("router_reasons") or []),
                                "router_id": h.get("router_id"), "router_signals": h.get("router_signals"),
                                "coverage_label": h.get("coverage_label"), "coverage_reasons": h.get("coverage_reasons"),
                                "critical_state": h.get("critical_state"), "repetition_count": h.get("repetition_count"),
                                "b1_b2_tvd": h.get("b1_b2_tvd")},
                    "hard_categories": _categories(h, obs) if obs else [],
                    "b1_probs": h.get("b1_probs"), "b2_v1_probs": h.get("probs"), "b2_v1_top": h.get("b2_top_action"),
                    "b2_v1_counterfactual": h.get("b2_counterfactual"),
                    "features": h.get("features"), "features_recomputed_match": recomputed == h.get("features"),
                    "legal": legal_action_set(obs) if obs else [], "legal_pairs": _pairs((raw or {}).get("legal_actions")),
                    "unsupported": (["enter_safe_code"] if obs and "enter_safe_code" in legal_action_set(obs)
                                    and schema.code_from_visible_puzzle(obs) is None else []),
                    "b2_v1_proposed": h.get("b2_proposed_action"), "history_tail": list(history[actor][-4:]),
                    "intended_action": validated.get("action"), "intended_target": validated.get("target_id"),
                    "observation_key": observation_key(policy_key, obs, _pairs((raw or {}).get("legal_actions"))) if obs else None,
                    "observation_hash": (raw or {}).get("observation_hash"), "recovery_outcome": (raw or {}).get("recovery_outcome"),
                    "exclusions_clean": hard + soft, "clean": not hard and not soft,
                    "in_frozen_escalation_dataset": None if frozen_ids is None else h.get("llm_decision_id") in frozen_ids,
                    "label_source": "llm_decision.validated_action", "feature_source": "hybrid_decision.features",
                }
                out.append(rec)
                if h.get("llm_outcome") == "committed":
                    history[actor].append(((raw or {}).get("executed_action") or {}).get("action") or h.get("sampled_action") or "wait")
            else:
                history[actor].append(h.get("sampled_action") or "wait")
    ids = {r["llm_decision_id"] for r in out}
    stats.update({"matches": len(dirs), "committed": len(out), "clean": sum(1 for r in out if r["clean"]),
                  "features_recomputed_match": sum(1 for r in out if r["features_recomputed_match"])})
    if frozen_ids is not None:
        stats["frozen_dataset_rows"] = len(frozen_ids)
        stats["frozen_dataset_equal"] = ids == frozen_ids
    return out, dict(stats)


# -- development sets ----------------------------------------------------------------------------------------

def _real_row(r: dict[str, Any], source: str, level: str) -> dict[str, Any]:
    pairs = _pairs(r.get("legal_actions"))
    return {"source": source, "decision_id": r["decision_id"], "match_id": r["match_id"], "seed": r["seed"], "level": level,
            "policy_key": r["policy_key"], "actor_id": r["actor_id"], "decision_number": r.get("decision_number"),
            "features": r["features"], "legal": legal_action_set(r["observation"]),
            "unsupported": (["enter_safe_code"] if "enter_safe_code" in legal_action_set(r["observation"])
                            and schema.code_from_visible_puzzle(r["observation"]) is None else []),
            "intended_action": r["intended_action"],
            "intended_target": r.get("intended_target"), "observation_key": observation_key(r["policy_key"], r["observation"], pairs),
            "coverage": r.get("coverage")}


def dev_real_decisions() -> tuple[list[dict], dict[str, Any]]:
    """CLEAN REAL decisions of the DEVELOPMENT configurations (M4D 8003 / 8004, Stage-4B 9008), read only."""
    from workshop.policy_emulator.features import PolicyFeatureExtractorV1
    out, stats = [], {}
    m4d = [r for r in _jsonl(M4D_REAL) if r["clean"] and role_of("m4d", r["seed"]) == "development"]
    out += [_real_row(r, "m4d_real", "level_01") for r in m4d]
    stats["m4d_real"] = {"path": str(M4D_REAL.relative_to(PROJECT_ROOT)), "sha256": _sha(M4D_REAL), "rows": len(m4d)}
    cfg = {"full_match": {"experiment_id": "stage4b-provider-pilot:real_reference", "root": STAGE4B_REAL_EXPORT,
                          "decisions": "decisions.jsonl", "manifest": "manifest.json"}}
    recs, sources = ds.normalize_full_match(cfg, "stage5a-dev-stage4b-real", PolicyFeatureExtractorV1(), PROJECT_ROOT)
    s4b = [r for r in recs if r["clean"] and role_of("stage4b", r["seed"]) == "development"]
    out += [_real_row(r, "stage4b_real", _level(r["match_id"], None)) for r in s4b]
    stats["stage4b_real"] = {"path": f"{STAGE4B_REAL_EXPORT}/decisions.jsonl", "sha256": sources["decisions"]["sha256"],
                             "rows": len(s4b), "all_records": len(recs)}
    return out, stats


def replay_states(source: str, role: str, *, allow_protected: bool = False) -> tuple[list[dict], dict[str, Any]]:
    """Every decision of the B2-through-R0 arm of the configurations with this role: what R2.1 would have seen."""
    if source == "stage4c":
        if not allow_protected:
            raise WorkshopError("STOP: Stage-4C replay states are read only by the regression step")
        dirs, level = sorted((PROJECT_ROOT / "workshop_runs/stage4c-confirmatory/matches/r0").glob("*")), None
    else:
        cfg = R0_SOURCES[source]
        dirs = sorted(p for p in PROJECT_ROOT.glob(cfg["match_dirs"]) if role_of(source, _seed(p.name)) == role)
        level = cfg["level"]
        for d in dirs:
            _guard_protected(d)
    out = []
    for d in dirs:
        if not (d / "telemetry_match01.jsonl").is_file():
            continue
        for e in _jsonl(d / "telemetry_match01.jsonl"):
            if e.get("event_type") != "hybrid_decision":
                continue
            out.append({"source": source, "config_id": d.name, "level": _level(d.name, level), "seed": _seed(d.name),
                        "actor_id": e["actor_id"], "policy_key": e["policy"], "decision_index": e.get("decision_index"),
                        "features": e["features"], "legal": e["legal_actions"], "unsupported": e.get("unsupported_masked") or [],
                        "proposed": e.get("intended_action"), "committed": e.get("sampled_action") or "wait",
                        "coverage_label": e.get("coverage_label"), "logged_signals": {k: (e.get("router_signals") or {}).get(k)
                                                                                      for k in ("repetition_loop", "strong_b1_b2_disagreement")},
                        "logged_b1_b2_tvd": e.get("b1_b2_tvd")})
    return out, {"matches": len(dirs), "decisions": len(out)}


# -- dedupe / conflicts --------------------------------------------------------------------------------------

def _entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return round(-sum(c / n * math.log2(c / n) for c in counts.values() if c), 4) if n else 0.0


def dedupe(hard: list[dict], phase_a: list[dict]) -> dict[str, Any]:
    """Apply the frozen dedupe rule; mutates hard[i]['status']. Returns counts and the conflict record."""
    pa_label: dict[str, set] = defaultdict(set)
    for r in phase_a:
        pa_label[observation_key(r["policy_key"], r["observation"], _pairs(r.get("legal_actions")))].add(r["intended_action"])
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)  # (role, key): dedupe within a role
    for r in hard:
        if not r["clean"]:
            r["status"] = "excluded: " + ", ".join(r["exclusions_clean"])
        elif r["role"] not in ("train", "development"):
            r["status"] = "excluded: configuration outside TRAIN / DEVELOPMENT"
        else:
            by_key[(r["role"], r["observation_key"])].append(r)
    conflicts = []
    for (_, key), group in by_key.items():
        labels = Counter(r["intended_action"] for r in group)
        if key in pa_label:
            for r in group:
                if pa_label[key] == {r["intended_action"]}:
                    r["status"] = "duplicate of a Phase-A CLEAN row (same label)"
                else:
                    r["status"] = "conflict with a Phase-A CLEAN row (different label): excluded"
            if any(pa_label[key] != {r["intended_action"]} for r in group):
                conflicts.append({"key": key, "with": "phase_a", "labels": dict(labels), "phase_a_labels": sorted(pa_label[key]),
                                  "entropy": _entropy(labels + Counter({a: 1 for a in pa_label[key]})),
                                  "sources": sorted({r["source"] for r in group})})
            continue
        if len(labels) > 1:
            for r in group:
                r["status"] = "conflicting REAL labels: excluded"
            conflicts.append({"key": key, "with": "hard examples", "labels": dict(labels), "entropy": _entropy(labels),
                              "sources": sorted({r["source"] for r in group}), "roles": sorted({r["role"] for r in group})})
            continue
        group[0]["status"] = "kept"
        for r in group[1:]:
            r["status"] = "duplicate hard example (same label): dropped"
    # leakage: a DEVELOPMENT key that TRAIN also contains
    train_keys = set(pa_label) | {r["observation_key"] for r in hard if r.get("status") == "kept" and r["role"] == "train"}
    leaks = 0
    for r in hard:
        if r.get("status") == "kept" and r["role"] == "development" and r["observation_key"] in train_keys:
            r["status"] = "development key also in TRAIN: excluded from development"
            leaks += 1
    return {"status_counts": dict(Counter(r["status"] for r in hard)), "conflicts": conflicts,
            "conflicting_keys": len(conflicts), "conflicting_records": sum(sum(c["labels"].values()) for c in conflicts),
            "development_leaks_excluded": leaks, "train_keys": train_keys}


# -- materialization -----------------------------------------------------------------------------------------

def materialize() -> dict[str, Any]:
    pol, split = load_policy(), load_split()
    if DATA_MANIFEST.is_file():
        raise WorkshopError(f"{DATA_MANIFEST} exists; the Stage-5A data are materialized once")
    phase_a = ds.load("phase-a-v2", "decisions_clean.jsonl")
    hard, extraction = [], {}
    for src in ESCALATION_SOURCES:
        rows, stats = extract_escalations(src)
        hard += rows
        extraction[src] = stats
    dd = dedupe(hard, phase_a)
    train_keys = dd.pop("train_keys")
    dev_real, dev_real_stats = dev_real_decisions()
    leak_real = [r for r in dev_real if r["observation_key"] in train_keys]
    dev_real = [r for r in dev_real if r["observation_key"] not in train_keys]
    dev_replay = []
    replay_stats = {}
    for src in R0_SOURCES:
        rows, st = replay_states(src, "development")
        dev_replay += rows
        replay_stats[src] = st
    files = {"hard_examples.jsonl": hard, "dev_real_decisions.jsonl": dev_real, "dev_replay_r0.jsonl": dev_replay}
    for name, rows in files.items():
        _write_jsonl(DATA_DIR / name, rows)
    train_hard = [r for r in hard if r.get("status") == "kept" and r["role"] == "train"]
    dev_hard = [r for r in hard if r.get("status") == "kept" and r["role"] == "development"]
    body = {"version": "stage5a-data-v1", "promotion_policy_hash": pol["policy_hash"], "split_hash": split["split_hash"],
            "files": {n: {"path": str((DATA_DIR / n).relative_to(PROJECT_ROOT)), "sha256": _sha(DATA_DIR / n), "rows": len(r)}
                      for n, r in files.items()},
            "phase_a_clean": {"path": str(PHASE_A.relative_to(PROJECT_ROOT)), "sha256": _sha(PHASE_A), "rows": len(phase_a),
                              "dataset_hash": ds.load_provenance("phase-a-v2")["dataset_hash"]},
            "extraction": extraction, "dedupe": dd, "dev_real": {**dev_real_stats, "leaks_excluded": len(leak_real),
                                                                 "rows_kept": len(dev_real)},
            "dev_replay": replay_stats,
            "counts": {"train_hard_examples": len(train_hard), "dev_hard_examples": len(dev_hard),
                       "dev_real_decisions": len(dev_real), "dev_replay_decisions": len(dev_replay)},
            "composition": composition(train_hard, dev_hard)}
    body = {**body, "data_hash": content_hash(body), "materialized_utc": _now()}
    _write_json(DATA_MANIFEST, body)
    return body


def load_data_manifest() -> dict[str, Any]:
    body = json.loads(DATA_MANIFEST.read_text())
    if content_hash({k: v for k, v in body.items() if k not in ("data_hash", "materialized_utc")}) != body["data_hash"]:
        raise WorkshopError(f"{DATA_MANIFEST}: hash mismatch")
    for name, f in body["files"].items():
        if _sha(PROJECT_ROOT / f["path"]) != f["sha256"]:
            raise WorkshopError(f"{f['path']} differs from the data manifest")
    return body


def load_rows(name: str) -> list[dict]:
    return _jsonl(DATA_DIR / name)


def composition(train_hard: list[dict], dev_hard: list[dict]) -> dict[str, Any]:
    def comp(rows: list[dict]) -> dict[str, Any]:
        return {"rows": len(rows), "by_source": dict(Counter(r["source"] for r in rows)),
                "by_level": dict(Counter(r["level"] for r in rows)), "by_policy": dict(sorted(Counter(r["policy_key"] for r in rows).items())),
                "by_model": dict(Counter(r["model"] for r in rows)), "by_personality": dict(Counter(r["personality"] for r in rows)),
                "by_trigger": dict(Counter(r["trigger"]["reason_class"] for r in rows)),
                "by_category": dict(Counter(c for r in rows for c in r["hard_categories"])),
                "by_coverage": dict(Counter(r["trigger"]["coverage_label"] for r in rows)),
                "labels": dict(Counter(r["intended_action"] for r in rows)),
                "b2_v1_top_equals_real": sum(1 for r in rows if r["b2_v1_top"] == r["intended_action"])}
    return {"train": comp(train_hard), "development": comp(dev_hard)}


# -- the registry ------------------------------------------------------------------------------------------

def _tree(pattern: str) -> dict[str, Any]:
    from workshop.policy_emulator import stage3_release as rel
    tele = sorted(PROJECT_ROOT.glob(pattern + "/telemetry_match01.jsonl"))
    return {"matches": len(tele), "telemetry_fingerprint": rel.tree_fingerprint(tele) if tele else None}


def build_registry() -> dict[str, Any]:
    pol, split, data = load_policy(), load_split(), load_data_manifest()
    s3 = json.loads((PROJECT_ROOT / "workshop/policy_data/stage3_data_registry.json").read_text())
    s4 = json.loads((PROJECT_ROOT / "workshop/policy_data/stage4_release_registry.json").read_text())
    s3_status = {e["path"]: e.get("status") for e in s3["entries"]}
    s4_class = {e["id"]: e for e in s4["entries"]}
    hard = load_rows("hard_examples.jsonl")
    comp_all = {}
    for src in ESCALATION_SOURCES:
        for role in ("train", "development"):
            rows = [r for r in hard if r["source"] == src and r["role"] == role and r.get("status") == "kept"]
            if rows:
                comp_all[(src, role)] = rows

    def esc_entry(src: str, role: str, seeds: list[int]) -> dict[str, Any]:
        cfg = ESCALATION_SOURCES[src]
        rows = [r for r in hard if r["source"] == src and r["seed"] in seeds]
        kept = comp_all.get((src, role), [])
        return {"id": f"{src}_escalations_{role}", "source_path": cfg["match_dirs"], "frozen_dataset": cfg["frozen_dataset"],
                "source_hash": _tree(cfg["match_dirs"])["telemetry_fingerprint"],
                "frozen_dataset_sha256": _sha(PROJECT_ROOT / cfg["frozen_dataset"]) if cfg["frozen_dataset"] else None,
                "source_stage": cfg["stage"], "original_role": "evaluation-only", "stage5_role": role, "seeds": seeds,
                "row_count": len(rows), "rows_promoted": len(kept), "configurations": len({r["config_id"] for r in rows}),
                "real_label": True, "models": sorted({r["model"] for r in rows}),
                "personalities": sorted({r["personality"] for r in rows}), "levels": sorted({r["level"] for r in rows}),
                "router_triggers": dict(Counter(r["trigger"]["reason_class"] for r in rows)),
                "eligibility": "eligible: REAL label, full observation, legal set, valid action, predates Stage 4C",
                "exclusion_reason": dict(Counter(r["status"] for r in rows if r.get("status") != "kept")) or None,
                "promotion_timestamp": pol["frozen_utc"]}

    def plain(eid: str, path: str, stage: str, original: str, role: str, why: str, **extra: Any) -> dict[str, Any]:
        if role not in ROLES:
            raise WorkshopError(role)
        return {"id": eid, "source_path": path, "source_stage": stage, "original_role": original, "stage5_role": role,
                "eligibility": "eligible" if role in ("train", "development") else "not eligible", "exclusion_reason": why
                if role not in ("train", "development") else None,
                "promotion_timestamp": pol["frozen_utc"] if role in ("train", "development") else None, **extra}
    m4d_split = SPLIT["train"]["m4d"], SPLIT["development"]["m4d"]
    b4_split = SPLIT["train"]["stage4b"], SPLIT["development"]["stage4b"]
    pa = data["phase_a_clean"]
    entries = [
        plain("phase_a_clean", pa["path"], "Stage 3 Phase A", s3_status.get(pa["path"]), "train", None,
              source_hash=pa["sha256"], row_count=pa["rows"], configurations=48, real_label=True,
              models=["gpt61sol", "sonnet55"], personalities=["cautious", "opportunist", "racer", "saboteur"],
              levels=["level_01"], router_triggers=None, note="the B2-v1 training set, unchanged"),
        plain("phase_a_all_valid", "workshop/policy_data/phase-a-v2/decisions_all_valid.jsonl", "Stage 3 Phase A",
              s3_status.get("workshop/policy_data/phase-a-v2/decisions_all_valid.jsonl"), "excluded",
              "superset of CLEAN with non-clean rows (the frozen clean rule)"),
        plain("stage2_controlled_snapshots", "workshop/policy_data/phase-a-v2/controlled_snapshot_decisions.jsonl", "Stage 2",
              s3_status.get("workshop/policy_data/phase-a-v2/controlled_snapshot_decisions.jsonl"), "excluded",
              "8 controlled snapshot states of a TRAIN configuration (seed 7303), not trajectory data; kept for the "
              "frozen Stage-3 controlled checks"),
        esc_entry("m4b", "train", SPLIT["train"]["m4b"]), esc_entry("m4c", "train", SPLIT["train"]["m4c"]),
        esc_entry("m4d", "train", m4d_split[0]), esc_entry("m4d", "development", m4d_split[1]),
        esc_entry("stage4b", "train", b4_split[0]), esc_entry("stage4b", "development", b4_split[1]),
        plain("m4d_real_train_configurations", "workshop/policy_data/hybrid-r21-confirmatory-real-v1/decisions_clean.jsonl "
              "(seeds 8001, 8002)", "Stage 3 M4D", "evaluation-only", "excluded",
              "configurations in the TRAIN partition; on-policy REAL trajectories are not hard examples under this "
              "policy, and as DEVELOPMENT they would share configurations with TRAIN"),
        plain("m4d_real_development", "workshop/policy_data/hybrid-r21-confirmatory-real-v1/decisions_clean.jsonl "
              "(seeds 8003, 8004)", "Stage 3 M4D", "evaluation-only", "development", None, source_hash=_sha(M4D_REAL),
              row_count=data["dev_real"]["m4d_real"]["rows"], real_label=True, levels=["level_01"]),
        plain("stage4b_real_train_configurations", f"{STAGE4B_REAL_EXPORT}/decisions.jsonl (seed 9007)", "Stage 4B",
              s4_class["stage4b_real_reference_export"]["class"], "excluded", "configurations in the TRAIN partition"),
        plain("stage4b_real_development", f"{STAGE4B_REAL_EXPORT}/decisions.jsonl (seed 9008)", "Stage 4B",
              s4_class["stage4b_real_reference_export"]["class"], "development", None,
              source_hash=data["dev_real"]["stage4b_real"]["sha256"], row_count=data["dev_real"]["stage4b_real"]["rows"],
              real_label=True, levels=["level_02", "level_03"]),
        plain("m4d_r0_development", "workshop_runs/stage3-hybrid-r21-confirmatory/r0_control (seeds 8003, 8004)", "Stage 3 M4D",
              "evaluation-only", "development", None, real_label=False, row_count=data["dev_replay"]["m4d"]["decisions"],
              use="R2.1 replay states only (B2-through-R0 trajectories; no label)"),
        plain("stage4b_r0_development", "workshop_runs/stage4b-provider-pilot/matches/r0 (seed 9008)", "Stage 4B",
              s4_class["stage4b_b2_through_r0"]["class"], "development", None, real_label=False,
              row_count=data["dev_replay"]["stage4b"]["decisions"], use="R2.1 replay states only"),
        plain("r0_b2_surrogate_arms_other", "M4B / M4C / M4D (TRAIN seeds) r0_control and b2 arms, Stage-4B r0 (9007), "
              "M3 closed loop", "Stage 3 / 4B", "evaluation-only", "excluded", "no REAL label"),
        plain("m4d_failed_epoch1", "workshop_runs/stage3-hybrid-r21-confirmatory/failed_attempts/epoch1", "Stage 3 M4D",
              s3_status.get("workshop_runs/stage3-hybrid-r21-confirmatory/failed_attempts/epoch1"), "invalid",
              "invalid infrastructure (expired session); excluded by path"),
        plain("stage4a_call_free", "workshop_runs/stage4a-shift-diagnostics(-rerun)", "Stage 4A", "evaluation-only", "excluded",
              "call-free: no REAL label"),
        plain("stage4a_first_run_report", "workshop/policy_reports/stage4/stage4a_first_run_report", "Stage 4A",
              "invalid-excluded", "invalid", "superseded report"),
        plain("engineering_sessions", "M3 pilot repro, M4A smoke, Stage-4A smoke / regression / R0 diagnosis, Stage-4B "
              "preflights", "Stage 3 / 4", "engineering-check-only", "engineering-only", "engineering checks"),
        plain("stage4c_dryrun", "workshop_runs/stage4c-prelaunch-dryrun", "Stage 4C prelaunch",
              s4_class["stage4c_dryrun"]["class"], "engineering-only", "fake-provider dry run"),
        plain("stage4c_dryrun_attempt01", "workshop_runs/stage4c-prelaunch-dryrun-attempt01", "Stage 4C prelaunch",
              s4_class["stage4c_dryrun_attempt01"]["class"], "invalid", "archived invalid attempt"),
    ]
    for eid in ("stage4c_real", "stage4c_real_reference_export", "stage4c_hybrid", "stage4c_escalations",
                "stage4c_b2_through_r0", "stage4c_bootstrap_output"):
        e = s4_class[eid]
        entries.append(plain(eid, e["path"], "Stage 4C (confirmatory)", e["class"], "regression-only",
                             "protected confirmatory data: regression guard after the candidate freeze only; never "
                             "training, development, hyperparameter, feature, threshold or router selection",
                             source_hash=e.get("telemetry_fingerprint") or e.get("sha256"), row_count=e.get("records") or e.get("matches"),
                             metadata_from="workshop/policy_data/stage4_release_registry.json (Stage-4C files not opened)"))
    entries.append(plain("stage4c_preflight", s4_class["stage4c_preflight"]["path"], "Stage 4C", "engineering-check-only",
                         "engineering-only", "credential / model-access probes"))
    body = {"version": "stage5-data-promotion-registry-v1", "promotion_policy_hash": pol["policy_hash"],
            "split_hash": split["split_hash"], "data_hash": data["data_hash"], "roles": list(ROLES), "entries": entries,
            "counts": data["counts"], "composition": data["composition"], "dedupe": data["dedupe"],
            "training_registry_unchanged": sorted(ds.DATASETS),
            "note": "Stage-5A promotion creates new files under workshop/policy_data/stage5a only; dataset.DATASETS and "
                    "every historical registry are unchanged"}
    return {**body, "registry_hash": content_hash(json.loads(json.dumps(body, default=str)))}


__all__ = ["POLICY", "SPLIT", "write_policy", "load_policy", "write_split", "load_split", "extract_escalations",
           "dev_real_decisions", "replay_states", "dedupe", "materialize", "load_data_manifest", "load_rows",
           "build_registry", "observation_key", "role_of", "ROLES", "REGISTRY_PATH", "DATA_DIR", "STAGE5_DIR"]
