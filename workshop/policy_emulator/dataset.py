"""3A - the canonical normalized Stage 3 dataset.

Sources (read only, never rewritten):

* Phase A (full matches): the exported `analysis/decisions.jsonl` gives the
  policy decisions and their labels; each record is joined by `decision_id`
  to its raw `llm_decision` telemetry event, which holds the full annotated
  obs-v1 observation and the legal actions with their targets (the export
  keeps action names only). The barrier commit, provider calls and action
  lifecycle of the same decision come from the same telemetry file.
* Stage 2 (controlled snapshots): the exported `analysis/decisions.jsonl`.
  Its records carry no payload, so each snapshot is rebuilt call-free with
  the Stage-2 builder (`behavior_grid_snapshots.build_frozen`, as its dry run
  does) and the rebuilt observation is checked against the logged
  `observation_hash`.

`source_type` is assigned here (full_match / controlled_snapshot): the
canonical dataset is where it becomes mandatory.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from workshop.policy_emulator import schema
from workshop.policy_emulator.features import PolicyFeatureExtractorV1, feature_schema
from workshop.runners.common import PROJECT_ROOT, display_path

DATASETS: dict[str, dict[str, Any]] = {
    "phase-a-v2": {
        "full_match": {
            "experiment_id": "phase-a-v2-sonnet55-vs-gpt61sol",
            "root": "workshop_runs/cross-model-phase-a-v2-sonnet55-vs-gpt61sol",
            "decisions": "analysis/decisions.jsonl",
            "manifest": "manifest.json",
        },
        "controlled_snapshot": {
            "experiment_id": "behavior-grid-stage2-v1-sonnet55-vs-gpt61sol",
            "root": "workshop_runs/behavior-grid-stage2-v1-sonnet55-vs-gpt61sol",
            "decisions": "analysis/decisions.jsonl",
            "config": "workshop/labs/controlled_behavior_grid/config.json",
        },
        "output": "workshop/policy_data/phase-a-v2",
    },
}

# what the canonical `observation` keeps of the logged obs-v1 payload: the model never sees
# `self.knowledge`, `communication` (off), `schema_version` or `time` (world-clock values
# under the barrier; time_budget carries the mission time it does see)
DROPPED_OBSERVATION_KEYS = ("communication", "time", "schema_version")
DROPPED_SELF_KEYS = ("knowledge",)

# Phase-A export field -> canonical field (the export's actual names, spec section 3)
PHASE_A_FIELD_MAP = {
    "action": "intended_action", "target": "intended_target", "fallback": "fallback_used", "repair": "repair_used",
    "legal": "legal_action_names (targets from the raw llm_decision.legal_actions)", "spy_id": "actor_id",
    "model": "lab_model_key -> model (sonnet55 / gpt61sol)", "model_id": "model_id", "personality": "personality",
    "seed": "seed", "side": "side", "match_id": "match_id", "decision_id": "decision_id", "valid": "valid",
    "provider": "provider", "category": "phase_a_category",
}
STAGE2_FIELD_MAP = {
    "intended_action": "intended_action", "intended_target": "intended_target", "intended_trap_type": "intended_trap_type",
    "committed_action": "committed_action", "committed_target": "committed_target", "fallback_used": "fallback_used",
    "repair_used": "repair_used", "clean": "clean (Stage-2 definition)", "model": "lab_model_key -> model",
    "personality": "personality", "snapshot_id": "snapshot_id", "category": "stage2_category",
    "legal_actions": "legal actions (targets from the rebuilt payload; checked equal)",
}
UNAVAILABLE_FIELDS = {
    "full_match": {
        "strategy_memory": "not logged in Phase-A telemetry (only its size and counts); recent actions are rebuilt "
                           "from the actor's earlier decisions in the match",
        "system_prompt_hash": "not logged in Phase-A telemetry (prompt_version and personality_profile are)",
        "committed_target_after_lifecycle": "the barrier commits the executed action; the lifecycle records how it ended",
    },
    "controlled_snapshot": {
        "match_id": "a frozen snapshot is not a match",
        "observation": "not in the export; rebuilt call-free from the snapshot definition and checked against observation_hash",
        "runtime_name": "not in the Stage-2 export (agent_runtime_version and representation_variant are)",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def observation_hash(payload: dict[str, Any]) -> str:
    """The LLM controller's own observation_hash (sha256 of the sorted JSON)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def project_observation(payload: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in payload.items() if k not in DROPPED_OBSERVATION_KEYS}
    if isinstance(out.get("self"), dict):
        out["self"] = {k: v for k, v in out["self"].items() if k not in DROPPED_SELF_KEYS}
    return out


def _legal_pairs(legal_actions: Iterable[dict]) -> set[tuple[str, str | None]]:
    return {(a.get("action"), a.get("target_id")) for a in legal_actions if isinstance(a, dict)}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -- Phase A --------------------------------------------------------------------------------------

@dataclass
class MatchTelemetry:
    run_dir: Path
    run_id: str
    game_version: str | None
    decisions: dict[str, dict]
    commits: dict[str, dict]
    lifecycles: dict[str, dict]
    calls: dict[str, list[dict]]
    recent: dict[str, list[str]]  # decision_id -> this actor's earlier selected actions, most recent first


def read_match_telemetry(run_dir: Path) -> MatchTelemetry:
    events = []
    for path in sorted(run_dir.glob("telemetry_match*.jsonl")):
        events.extend(_read_jsonl(path))
    decisions = {e["decision_id"]: e for e in events if e.get("event_type") == "llm_decision" and e.get("decision_id")}
    commits = {e["decision_id"]: e for e in events if e.get("event_type") in ("barrier_commit", "live_async_commit")
               and e.get("decision_id")}
    lifecycles = {e["decision_id"]: e for e in events if e.get("event_type") == "action_lifecycle" and e.get("decision_id")}
    calls: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("event_type") == "provider_call" and e.get("decision_id"):
            calls[e["decision_id"]].append(e)
    recent: dict[str, list[str]] = {}
    by_actor: dict[str, list[dict]] = defaultdict(list)
    for e in decisions.values():
        by_actor[e.get("actor_id")].append(e)
    for actor_events in by_actor.values():
        history: list[str] = []
        for e in sorted(actor_events, key=lambda x: (x.get("decision_number") or 0, x.get("tick") or 0)):
            recent[e["decision_id"]] = list(reversed(history))[:5]
            # what the agent's memory records: the selected action, a fallback as wait
            selected = (e.get("validated_action") or {}).get("action") if e.get("final_model_action_valid") else "wait"
            history.append(selected or "wait")
    run_json = run_dir / "run.json"
    meta = json.loads(run_json.read_text()) if run_json.is_file() else {}
    return MatchTelemetry(run_dir, meta.get("run_id") or run_dir.name, meta.get("game_version"), decisions, commits,
                          lifecycles, dict(calls), recent)


def full_match_exclusions(raw: dict | None, commit: dict | None) -> tuple[list[str], list[str]]:
    """(reasons excluded from ALL_VALID, reasons excluded from CLEAN only)."""
    hard: list[str] = []
    soft: list[str] = []
    if raw is None or not isinstance(raw.get("observation"), dict) or not raw.get("legal_actions"):
        return ["telemetry_incomplete"], []
    if commit is None:
        hard.append("telemetry_incomplete")
    if raw.get("fallback_used") or not raw.get("final_model_action_valid"):
        hard.append("fallback")
    else:
        chosen = raw.get("validated_action") or {}
        if (chosen.get("action"), chosen.get("target_id")) not in _legal_pairs(raw["legal_actions"]):
            hard.append("action_not_in_legal_set")
    if raw.get("repair_attempted"):
        error = str(raw.get("initial_error_type") or "")
        soft.append("repair_after_provider_error" if error.startswith("provider_") else "repair_after_invalid_output")
    return hard, soft


def normalize_full_match(cfg: dict[str, Any], dataset: str, extractor: PolicyFeatureExtractorV1,
                         project_root: Path = PROJECT_ROOT) -> tuple[list[dict], dict[str, Any]]:
    src = cfg["full_match"]
    root = project_root / src["root"]
    manifest = json.loads((root / src["manifest"]).read_text())
    exported = _read_jsonl(root / src["decisions"])
    runs = {m["name"]: m for m in manifest.get("matches", []) if m.get("status") == "done"}
    telemetry: dict[str, MatchTelemetry] = {}
    records, missing_join = [], 0
    for row in exported:
        match = runs.get(row["match_id"])
        if match is None:
            missing_join += 1
            continue
        if row["match_id"] not in telemetry:
            telemetry[row["match_id"]] = read_match_telemetry(project_root / match["run_dir"])
        t = telemetry[row["match_id"]]
        raw = t.decisions.get(row["decision_id"])
        commit = t.commits.get(row["decision_id"])
        hard, soft = full_match_exclusions(raw, commit)
        model = schema.MODELS[row["model"]]
        payload = raw.get("observation") if raw else None
        validated = (raw or {}).get("validated_action") or {}
        executed = (raw or {}).get("executed_action") or {}
        lifecycle = t.lifecycles.get(row["decision_id"]) or {}
        calls = t.calls.get(row["decision_id"], [])
        recent = t.recent.get(row["decision_id"], [])
        error = str((raw or {}).get("initial_error_type") or "")
        record = {
            "source_type": schema.FULL_MATCH,
            "canonical_schema_version": schema.CANONICAL_SCHEMA_VERSION,
            "dataset": dataset, "experiment_id": src["experiment_id"], "run_id": t.run_id,
            "match_id": row["match_id"], "seed": row["seed"], "side": row["side"], "actor_id": row["spy_id"],
            "provider": row["provider"], "model_id": row["model_id"], "lab_model_key": row["model"], "model": model["name"],
            "personality": row["personality"], "policy_key": schema.policy_key(model["name"], row["personality"]),
            "policy_id": schema.policy_id(model["name"], row["personality"], dataset),
            "opponent_model": schema.MODELS.get(row.get("opponent_model"), {}).get("name"),
            "decision_id": row["decision_id"], "decision_number": (raw or {}).get("decision_number"),
            "mission_time_used_s": row.get("t"),
            "observation": project_observation(payload) if payload else None,
            "memory": {"observation_memory": (payload or {}).get("memory"), "recent_actions": recent,
                       "recent_actions_source": "rebuilt from this actor's earlier decisions in the match "
                                                "(strategy memory is not logged in Phase-A telemetry)"},
            "legal_actions": (raw or {}).get("legal_actions"),
            "legal_action_names": row.get("legal"),
            "intended_action": validated.get("action"), "intended_target": validated.get("target_id"),
            "intended_trap_type": validated.get("trap_type"), "intended_code_given": bool(validated.get("code")),
            "committed_action": executed.get("action") if commit else None,
            "committed_target": executed.get("target_id") if commit else None,
            "lifecycle_status": lifecycle.get("action_status"), "lifecycle_reason": lifecycle.get("action_terminal_reason"),
            "valid": bool(row.get("valid")), "fallback_used": bool(row.get("fallback")),
            "repair_used": bool(row.get("repair")), "initial_error_type": (raw or {}).get("initial_error_type"),
            "parser_failure": bool(raw) and not raw.get("initial_model_output_valid", True) and not error.startswith("provider_"),
            "provider_calls": len(calls),
            # transport retries with the same prompt (none in Phase A: the second call is a repair prompt)
            "provider_transport_retries": sum(1 for c in calls if (c.get("retry_count") or 0) > 0),
            "recovery_outcome": (raw or {}).get("recovery_outcome"),
            "phase_a_category": row.get("category"),
            "category": schema.category(validated.get("action"), validated.get("target_id"), payload or {}) if validated else None,
            "target_class": (schema.target_class(validated["action"], validated.get("target_id"), payload or {})
                             if validated.get("action") else None),
            "prompt_version": (raw or {}).get("prompt_version"),
            "observation_version": (raw or {}).get("observation_schema_version"),
            "memory_version": (raw or {}).get("memory_schema_version"),
            "action_schema_version": (raw or {}).get("action_schema_version"),
            "runtime_name": (raw or {}).get("agent_runtime"), "runtime_version": (raw or {}).get("agent_runtime_version"),
            "representation_variant": (raw or {}).get("representation_variant"),
            "game_code_hash": t.game_version or manifest.get("game_version"),
            "observation_hash": (raw or {}).get("observation_hash"),
            "observation_hash_verified": bool(payload) and observation_hash(payload) == (raw or {}).get("observation_hash"),
            "feature_schema_version": extractor.version,
            "features": extractor.extract(payload, recent) if payload else None,
            "exclusions_all_valid": hard, "exclusions_clean": hard + soft,
            "all_valid": not hard, "clean": not hard and not soft,
        }
        records.append(record)
    sources = {"decisions": {"path": f"{src['root']}/{src['decisions']}", "sha256": sha256_file(root / src["decisions"]),
                             "records": len(exported)},
               "manifest": {"path": f"{src['root']}/{src['manifest']}", "sha256": sha256_file(root / src["manifest"])},
               "telemetry": {f"{display_path(t.run_dir)}/{p.name}": sha256_file(p)
                             for t in sorted(telemetry.values(), key=lambda x: x.run_id)
                             for p in sorted(t.run_dir.glob("telemetry_match*.jsonl"))},
               "matches": len(telemetry), "export_rows_without_match": missing_join,
               "game_version": manifest.get("game_version")}
    return records, sources


# -- Stage 2 --------------------------------------------------------------------------------------

def rebuild_snapshot_payloads(snapshot_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Each Stage-2 snapshot's annotated payload and fixture memory, rebuilt call-free
    (StubClient; no sample is written - the frozen grid is never sampled again)."""
    from src.mission_time import annotate_live_payload
    from workshop.scripts import behavior_grid as grid
    from workshop.scripts import behavior_grid_snapshots as bgs

    cfg = grid.load_config()
    out = {}
    for snapshot_id in snapshot_ids:
        snapshot = cfg["_snapshots"][snapshot_id]
        model = next(iter(cfg["_models"].values()))
        with contextlib.redirect_stdout(io.StringIO()):
            frozen = bgs.build_frozen(snapshot, cfg["personalities"][0], lab=cfg, model=model,
                                      client_factory=lambda spec: grid.StubClient(spec))
        payload = annotate_live_payload(frozen.observation, frozen.payload)
        memory = frozen.controller.memory.to_compact_dict()
        decisions = memory.get("recent_decisions") or []
        recent = [d.get("selected_action") for d in reversed(decisions) if isinstance(d, dict)]
        out[snapshot_id] = {"payload": payload, "recent_actions": [a for a in recent if a][:5],
                            "strategy_memory": memory, "snapshot": snapshot, "game_seed": snapshot.get("game_seed"),
                            "facts": frozen.facts}
    return out


def normalize_controlled(cfg: dict[str, Any], dataset: str, extractor: PolicyFeatureExtractorV1,
                         project_root: Path = PROJECT_ROOT,
                         rebuilt: dict[str, dict[str, Any]] | None = None) -> tuple[list[dict], dict[str, Any]]:
    src = cfg["controlled_snapshot"]
    root = project_root / src["root"]
    exported = _read_jsonl(root / src["decisions"])
    rebuilt = rebuilt if rebuilt is not None else rebuild_snapshot_payloads(sorted({r["snapshot_id"] for r in exported}))
    records = []
    for row in exported:
        snap = rebuilt[row["snapshot_id"]]
        payload = snap["payload"]
        model = schema.MODELS[row["model"]]
        legal_from_payload = {a: sorted(t for t in ts if t) for a, ts in schema.legal_targets(payload).items()}
        legal_logged = {a: sorted(t for t in (ts or []) if t) for a, ts in (row.get("legal_actions") or {}).items()}
        hard = []
        if row.get("fallback_used") or not row.get("intended_action") or row.get("harness_error"):
            hard.append("fallback")
        soft = [] if row.get("clean") else ["not_clean_stage2"]
        records.append({
            "source_type": schema.CONTROLLED_SNAPSHOT,
            "canonical_schema_version": schema.CANONICAL_SCHEMA_VERSION,
            "dataset": dataset, "experiment_id": src["experiment_id"], "run_id": row.get("label"),
            "match_id": None, "seed": snap["game_seed"], "side": "white" if row["actor_id"] == "white_spy" else "black",
            "actor_id": row["actor_id"], "provider": row.get("provider"), "model_id": row.get("model_id"),
            "lab_model_key": row["model"], "model": model["name"], "personality": row["personality"],
            "policy_key": schema.policy_key(model["name"], row["personality"]),
            "policy_id": schema.policy_id(model["name"], row["personality"], dataset),
            "snapshot_id": row["snapshot_id"], "snapshot_code": row.get("snapshot_code"), "family": row.get("family"),
            "sample_index": row.get("sample_index"), "controlled_variables": row.get("controlled_variables"),
            "observation": project_observation(payload),
            "memory": {"observation_memory": payload.get("memory"), "recent_actions": snap["recent_actions"],
                       "recent_actions_source": "the snapshot's memory fixture (strategy_memory.recent_decisions)"},
            "legal_actions": payload.get("legal_actions"),
            "legal_actions_match_export": legal_from_payload == legal_logged,
            "intended_action": row.get("intended_action"), "intended_target": row.get("intended_target"),
            "intended_trap_type": row.get("intended_trap_type"),
            "committed_action": row.get("committed_action"), "committed_target": row.get("committed_target"),
            "valid": bool(row.get("valid")), "fallback_used": bool(row.get("fallback_used")),
            "repair_used": bool(row.get("repair_used")), "initial_error_type": row.get("initial_error_type"),
            "parser_failure": bool(row.get("parser_failure")), "provider_calls": row.get("provider_calls"),
            "provider_transport_retries": 0,
            "stage2_category": row.get("category"),
            "category": schema.category(row.get("intended_action"), row.get("intended_target"), payload),
            "target_class": (schema.target_class(row["intended_action"], row.get("intended_target"), payload)
                             if row.get("intended_action") else None),
            "prompt_version": row.get("prompt_version"), "observation_version": row.get("observation_version"),
            "memory_version": row.get("memory_version"), "action_schema_version": row.get("action_schema_version"),
            "runtime_name": None, "runtime_version": row.get("agent_runtime_version"),
            "representation_variant": row.get("representation_variant"),
            "game_code_hash": row.get("game_code_hash"),
            "observation_hash": row.get("observation_hash"),
            "observation_hash_verified": observation_hash(payload) == row.get("observation_hash"),
            "feature_schema_version": extractor.version,
            "features": extractor.extract(payload, snap["recent_actions"]),
            "exclusions_all_valid": hard, "exclusions_clean": hard + soft,
            "all_valid": not hard, "clean": not hard and not soft,
        })
    sources = {"decisions": {"path": f"{src['root']}/{src['decisions']}", "sha256": sha256_file(root / src["decisions"]),
                             "records": len(exported)},
               "config": {"path": src["config"], "sha256": sha256_file(project_root / src["config"])},
               "snapshots_rebuilt": sorted(rebuilt),
               "rebuild": "behavior_grid_snapshots.build_frozen with StubClient (no provider call, no sample written)"}
    return records, sources


# -- writing -------------------------------------------------------------------------------------

def _versions(records: list[dict]) -> dict[str, list]:
    keys = ("prompt_version", "observation_version", "memory_version", "action_schema_version", "runtime_name",
            "runtime_version", "game_code_hash")
    return {k: sorted({str(r.get(k)) for r in records}) for k in keys}


def summarize(full: list[dict], controlled: list[dict]) -> dict[str, Any]:
    def per_policy(rows):
        return dict(sorted(Counter(r["policy_key"] for r in rows).items()))
    clean = [r for r in full if r["clean"]]
    valid = [r for r in full if r["all_valid"]]
    return {
        "full_match": {
            "records": len(full), "all_valid": len(valid), "clean": len(clean),
            "excluded_from_all_valid": dict(Counter(x for r in full for x in r["exclusions_all_valid"])),
            "excluded_from_clean": dict(Counter(x for r in full for x in r["exclusions_clean"])),
            # one reason per excluded record (the first: fallback before repair)
            "excluded_from_clean_records": dict(Counter(r["exclusions_clean"][0] for r in full if r["exclusions_clean"])),
            "observation_hash_verified": sum(1 for r in full if r["observation_hash_verified"]),
            "provider_transport_retries_flagged": sum(1 for r in full if r["provider_transport_retries"]),
            "seeds": dict(sorted(Counter(r["seed"] for r in clean).items())),
            "matches": len({r["match_id"] for r in full}),
            "clean_per_policy": per_policy(clean), "all_valid_per_policy": per_policy(valid),
            "clean_actions": dict(Counter(r["intended_action"] for r in clean).most_common()),
            "clean_categories": dict(Counter(r["category"] for r in clean).most_common()),
        },
        "controlled_snapshot": {
            "records": len(controlled), "all_valid": sum(1 for r in controlled if r["all_valid"]),
            "clean": sum(1 for r in controlled if r["clean"]),
            "observation_hash_verified": sum(1 for r in controlled if r["observation_hash_verified"]),
            "legal_actions_match_export": sum(1 for r in controlled if r["legal_actions_match_export"]),
            "snapshots": sorted({r["snapshot_id"] for r in controlled}),
            "clean_per_policy": per_policy([r for r in controlled if r["clean"]]),
        },
        "policy_identities": schema.all_policy_keys(),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str, separators=(",", ":")) + "\n")
            n += 1
    return n


def build(dataset: str = "phase-a-v2", *, out_dir: Path | None = None, project_root: Path = PROJECT_ROOT,
          rebuilt: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    cfg = DATASETS[dataset]
    extractor = PolicyFeatureExtractorV1()
    full, full_sources = normalize_full_match(cfg, dataset, extractor, project_root)
    controlled, controlled_sources = normalize_controlled(cfg, dataset, extractor, project_root, rebuilt=rebuilt)
    out = out_dir or project_root / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "decisions_clean.jsonl": [r for r in full if r["clean"]],
        "decisions_all_valid.jsonl": [r for r in full if r["all_valid"]],
        "controlled_snapshot_decisions.jsonl": [r for r in controlled if r["all_valid"]],
    }
    for name, rows in files.items():
        write_jsonl(out / name, rows)
    (out / "feature_schema.json").write_text(json.dumps(feature_schema(), indent=2) + "\n")
    summary = summarize(full, controlled)
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    hashes = {name: sha256_file(out / name) for name in (*files, "feature_schema.json", "dataset_summary.json")}
    combined = hashlib.sha256("".join(f"{k}={v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()
    provenance = {
        "dataset": dataset, "dataset_hash": "sha256:" + combined[:16], "files": hashes,
        "canonical_schema_version": schema.CANONICAL_SCHEMA_VERSION,
        "normalization_code_version": schema.NORMALIZATION_VERSION,
        "normalization_code_hash": sha256_file(Path(__file__)),
        "feature_schema_version": extractor.version,
        "source_type_mapping": {cfg["full_match"]["experiment_id"]: schema.FULL_MATCH,
                                cfg["controlled_snapshot"]["experiment_id"]: schema.CONTROLLED_SNAPSHOT},
        "sources": {schema.FULL_MATCH: {"experiment_id": cfg["full_match"]["experiment_id"], **full_sources},
                    schema.CONTROLLED_SNAPSHOT: {"experiment_id": cfg["controlled_snapshot"]["experiment_id"],
                                                 **controlled_sources}},
        "field_mappings": {schema.FULL_MATCH: PHASE_A_FIELD_MAP, schema.CONTROLLED_SNAPSHOT: STAGE2_FIELD_MAP},
        "unavailable_fields": UNAVAILABLE_FIELDS,
        "observation_projection": {"dropped_top_level": list(DROPPED_OBSERVATION_KEYS),
                                   "dropped_self": list(DROPPED_SELF_KEYS),
                                   "why": "fields the model never sees (compact_observation_for_model + slim_live_payload)"},
        "clean_definition": {"excluded_from_all_valid": ["fallback", "action_not_in_legal_set", "telemetry_incomplete"],
                             "excluded_from_clean_also": ["repair_after_provider_error", "repair_after_invalid_output",
                                                          "not_clean_stage2"],
                             "flagged_not_excluded": ["provider_transport_retries"]},
        "versions": {schema.FULL_MATCH: _versions(full), schema.CONTROLLED_SNAPSHOT: _versions(controlled)},
        "originals_untouched": "sources are opened read-only; their sha256 is recorded above",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"out_dir": out, "summary": summary, "provenance": provenance}


def load(dataset: str = "phase-a-v2", name: str = "decisions_clean.jsonl", *, root: Path | None = None) -> list[dict]:
    base = root or PROJECT_ROOT / DATASETS[dataset]["output"]
    return _read_jsonl(base / name)


def load_provenance(dataset: str = "phase-a-v2", *, root: Path | None = None) -> dict[str, Any]:
    base = root or PROJECT_ROOT / DATASETS[dataset]["output"]
    return json.loads((base / "provenance.json").read_text())
