"""Policy artifacts: one directory per emulator, JSON only (nothing pickled).

    workshop/policies/<dataset>/<model>/<personality>/
        frequency-v1/  table-v1/  logistic-v1/  tree-v1/     action models (B0-B3)
        target-v1/     target-v2/   target-v2.1/             target policies
        coverage-v1/                                          coverage estimator

Each directory holds `model.json` (parameters), `metadata.json` (policy identity,
training dataset + hash, feature schema version, the training configuration,
the action vocabulary, the content hash of model.json) and, for action models,
`feature_schema.json` and `training_summary.json`. `load_policy` checks all of
it before a surrogate may play.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workshop.policy_emulator import schema
from workshop.policy_emulator.coverage import COVERAGE_VERSION, PolicyCoverageEstimator
from workshop.policy_emulator.features import FEATURE_SCHEMA_VERSION, feature_schema
from workshop.policy_emulator.models import LOGISTIC_VERSION, TREE_VERSION, LogisticPolicy, TreePolicy
from workshop.policy_emulator.policies import B1_DIMENSIONS, TARGET_DIMENSIONS, TablePolicy, TargetPolicy
from workshop.policy_emulator.targets import (CONTEXT_DIMENSIONS, TARGET_FEATURES_VERSION, TARGET_V21_VERSION,
                                              TARGET_V2_VERSION, TargetPolicyV2, TargetPolicyV21)
from workshop.runners.common import PROJECT_ROOT

POLICY_ROOT = "workshop/policies"
ARTIFACT_VERSION = "policy-artifact-v2"
TARGET_V1_VERSION = "target-v1"
ACTION_EMULATORS = (schema.FREQUENCY_POLICY_VERSION, schema.TABLE_POLICY_VERSION, LOGISTIC_VERSION, TREE_VERSION)
TARGET_VERSIONS = (TARGET_V1_VERSION, TARGET_V2_VERSION, TARGET_V21_VERSION)
DEFAULT_TARGET = TARGET_V2_VERSION
CONFIG_FILE = Path(__file__).with_name("model_configs.json")


class ArtifactProvenanceError(RuntimeError):
    """A policy artifact does not match the dataset / feature schema it claims."""


def model_configs() -> dict[str, dict[str, Any]]:
    """The fold-independent training configuration of every emulator (chosen in the
    leave-one-seed-out comparison, see the report's section O)."""
    return json.loads(CONFIG_FILE.read_text())["configs"]


def build_action_model(emulator: str, rows: list[dict], config: dict[str, Any] | None = None):
    cfg = dict(config if config is not None else model_configs().get(emulator, {}))
    if emulator == schema.FREQUENCY_POLICY_VERSION:
        return TablePolicy(dims=()).fit(rows)
    if emulator == schema.TABLE_POLICY_VERSION:
        return TablePolicy(dims=B1_DIMENSIONS, **cfg).fit(rows)
    if emulator == LOGISTIC_VERSION:
        return LogisticPolicy(**cfg).fit(rows)
    if emulator == TREE_VERSION:
        return TreePolicy(**cfg).fit(rows)
    raise ValueError(f"unknown emulator {emulator!r}; available: {', '.join(ACTION_EMULATORS)}")


def build_target(version: str, rows: list[dict], pooled_rows: list[dict]):
    if version == TARGET_V1_VERSION:
        return TargetPolicy().fit(rows, pooled_rows)
    if version == TARGET_V2_VERSION:
        return TargetPolicyV2().fit(rows, pooled_rows)
    if version == TARGET_V21_VERSION:
        return TargetPolicyV21().fit(rows, pooled_rows)
    raise ValueError(f"unknown target policy {version!r}; available: {', '.join(TARGET_VERSIONS)}")


def _from_json(kind: str, version: str, data: dict[str, Any]):
    if kind == "action":
        return (LogisticPolicy if version == LOGISTIC_VERSION else TreePolicy if version == TREE_VERSION
                else TablePolicy).from_json(data)
    if kind == "target":
        return (TargetPolicyV21 if version == TARGET_V21_VERSION else TargetPolicyV2 if version == TARGET_V2_VERSION
                else TargetPolicy).from_json(data)
    return PolicyCoverageEstimator.from_json(data)


def content_hash(data: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def policy_dir(dataset: str, policy_key: str, root: Path | None = None) -> Path:
    model, personality = schema.parse_policy_key(policy_key)
    return (root or PROJECT_ROOT / POLICY_ROOT) / dataset / model / personality


def _versions(rows: list[dict], key: str) -> list[str]:
    return sorted({str(r.get(key)) for r in rows})


def _identity(policy_key: str, dataset: str, rows: list[dict], provenance: dict[str, Any]) -> dict[str, Any]:
    model_name, personality = schema.parse_policy_key(policy_key)
    return {
        "artifact_version": ARTIFACT_VERSION, "policy_key": policy_key,
        "policy_id": schema.policy_id(model_name, personality, dataset), "model": model_name,
        "model_id": schema.MODEL_BY_NAME[model_name]["model_id"], "model_display": schema.MODEL_BY_NAME[model_name]["display"],
        "personality": personality, "prompt_version": _versions(rows, "prompt_version"),
        "observation_version": _versions(rows, "observation_version"), "memory_version": _versions(rows, "memory_version"),
        "action_schema_version": _versions(rows, "action_schema_version"), "runtime": _versions(rows, "runtime_name"),
        "game_code_hash": _versions(rows, "game_code_hash"), "training_dataset": dataset,
        "training_dataset_hash": provenance.get("dataset_hash"), "training_source_type": schema.FULL_MATCH,
        "training_set": "CLEAN", "training_records": len(rows),
        "training_seeds": sorted({r["seed"] for r in rows}), "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _write(out: Path, kind: str, version: str, model_json: dict[str, Any], identity: dict[str, Any],
           config: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model_json, indent=1, sort_keys=True) + "\n")
    meta = {**identity, "kind": kind, "emulator": version, "training_config": config,
            "action_vocabulary": list(schema.ACTIONS), "content_hash": content_hash(model_json), **(extra or {})}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def save_action(dataset: str, policy_key: str, emulator: str, rows: list[dict], provenance: dict[str, Any], *,
                config: dict[str, Any] | None = None, root: Path | None = None) -> Path:
    cfg = dict(config if config is not None else model_configs().get(emulator, {}))
    model = build_action_model(emulator, rows, cfg)
    out = policy_dir(dataset, policy_key, root) / emulator
    _write(out, "action", emulator, model.to_json(), _identity(policy_key, dataset, rows, provenance), cfg)
    (out / "feature_schema.json").write_text(json.dumps({**feature_schema(), "b1_dimensions": list(B1_DIMENSIONS)},
                                                        indent=2) + "\n")
    summary = {"policy_key": policy_key, "emulator": emulator, "training_records": len(rows),
               "seeds": dict(sorted(Counter(r["seed"] for r in rows).items())),
               "matches": len({r["match_id"] for r in rows}),
               "actions": dict(Counter(r["intended_action"] for r in rows).most_common()),
               "categories": dict(Counter(r["category"] for r in rows).most_common())}
    if isinstance(model, TablePolicy):
        summary["buckets_per_level"] = {str(k): len(v) for k, v in model.tables.items()}
    if isinstance(model, TreePolicy):
        summary.update({"depth": model.depth, "leaves": len(model.leaves), "rules": model.rules()})
    if isinstance(model, LogisticPolicy):
        summary.update({"columns": len(model.encoder.columns), "classes": model.classes, "optimizer": model.info})
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out


def save_target(dataset: str, policy_key: str, version: str, rows: list[dict], pooled_rows: list[dict],
                provenance: dict[str, Any], *, root: Path | None = None) -> Path:
    target = build_target(version, rows, pooled_rows)
    cfg = {"dims": list(TARGET_DIMENSIONS)} if version == TARGET_V1_VERSION else \
        {"c": target.c, "target_features_version": TARGET_FEATURES_VERSION}
    if version == TARGET_V21_VERSION:
        cfg.update({"prior": target.prior, "context_dimensions": list(CONTEXT_DIMENSIONS)})
    return _write(policy_dir(dataset, policy_key, root) / version, "target", version, target.to_json(),
                  _identity(policy_key, dataset, rows, provenance), cfg)


def save_coverage(dataset: str, policy_key: str, rows: list[dict], provenance: dict[str, Any], *,
                  root: Path | None = None) -> Path:
    est = PolicyCoverageEstimator().fit(rows)
    return _write(policy_dir(dataset, policy_key, root) / COVERAGE_VERSION, "coverage", COVERAGE_VERSION, est.to_json(),
                  _identity(policy_key, dataset, rows, provenance), {"b1_dimensions": list(B1_DIMENSIONS)})


def save_all(dataset: str, policy_key: str, rows: list[dict], pooled_rows: list[dict], provenance: dict[str, Any], *,
             emulators=ACTION_EMULATORS, root: Path | None = None) -> list[Path]:
    paths = [save_action(dataset, policy_key, e, rows, provenance, root=root) for e in emulators]
    paths += [save_target(dataset, policy_key, v, rows, pooled_rows, provenance, root=root) for v in TARGET_VERSIONS]
    paths.append(save_coverage(dataset, policy_key, rows, provenance, root=root))
    return paths


def _load(base: Path, kind: str, version: str, policy_key: str, expected_dataset_hash: str | None):
    try:
        metadata = json.loads((base / version / "metadata.json").read_text())
        data = json.loads((base / version / "model.json").read_text())
    except FileNotFoundError as exc:
        raise ArtifactProvenanceError(f"{kind} artifact {version} missing for {policy_key}: {exc.filename}. Train it: "
                                      "python -m workshop.scripts.train_policy_emulator --all-policies --model all") from None
    problems = []
    if metadata.get("policy_key") != policy_key:
        problems.append(f"metadata names {metadata.get('policy_key')}, not {policy_key}")
    if metadata.get("kind") != kind or metadata.get("emulator") != version:
        problems.append(f"metadata is {metadata.get('kind')} {metadata.get('emulator')}, not {kind} {version}")
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        problems.append(f"feature schema {metadata.get('feature_schema_version')} != {FEATURE_SCHEMA_VERSION}")
    if metadata.get("content_hash") != content_hash(data):
        problems.append("model.json does not match its recorded content hash")
    if expected_dataset_hash and metadata.get("training_dataset_hash") != expected_dataset_hash:
        problems.append(f"trained on dataset {metadata.get('training_dataset_hash')}, current is {expected_dataset_hash}")
    if problems:
        raise ArtifactProvenanceError(f"{base / version}: " + "; ".join(problems))
    return _from_json(kind, version, data), metadata


@dataclass
class LoadedPolicy:
    policy_key: str
    action_policy: Any
    target_policy: Any
    metadata: dict[str, Any]
    coverage: PolicyCoverageEstimator | None = None
    target_metadata: dict[str, Any] | None = None
    coverage_metadata: dict[str, Any] | None = None

    @property
    def emulator(self) -> str:
        return self.metadata["emulator"]

    @property
    def target_version(self) -> str:
        return (self.target_metadata or {}).get("emulator", "")

    @property
    def coverage_hash(self) -> str | None:
        return (self.coverage_metadata or {}).get("content_hash")


def load_policy(dataset: str, policy_key: str, *, emulator: str = schema.TABLE_POLICY_VERSION,
                target: str = DEFAULT_TARGET, coverage: bool = True, root: Path | None = None,
                expected_dataset_hash: str | None = None) -> LoadedPolicy:
    """One policy identity: its action model, its target policy and its coverage estimator,
    each provenance-checked (identity, kind / version, feature schema, content hash and,
    when given, the training dataset hash)."""
    if emulator not in ACTION_EMULATORS:
        raise ArtifactProvenanceError(f"unknown emulator {emulator!r}; available: {', '.join(ACTION_EMULATORS)}")
    if target not in TARGET_VERSIONS:
        raise ArtifactProvenanceError(f"unknown target policy {target!r}; available: {', '.join(TARGET_VERSIONS)}")
    base = policy_dir(dataset, policy_key, root)
    action, metadata = _load(base, "action", emulator, policy_key, expected_dataset_hash)
    target_policy, target_meta = _load(base, "target", target, policy_key, expected_dataset_hash)
    cov, cov_meta = _load(base, "coverage", COVERAGE_VERSION, policy_key, expected_dataset_hash) if coverage else (None, None)
    return LoadedPolicy(policy_key, action, target_policy, metadata, cov, target_meta, cov_meta)
