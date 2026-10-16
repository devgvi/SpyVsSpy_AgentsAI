"""M5D.1 spec section 32: shared seed enforcement, paired result alignment,
integrity checks, incompatible exclusion from gameplay aggregates, no
duplicate episode IDs, aggregate/raw reconciliation. All synthetic - no
live model, no real Game run.
"""

import json

import pytest

from src.evaluation.compatibility import CompatibilityStatus
from src.evaluation.episode_runner import EpisodeResult, EpisodeStatus
from src.evaluation.integrity import verify_integrity
from src.evaluation.paired import paired_seed_rows, render_paired_markdown_table
from src.evaluation.results import ExperimentResult
from src.evaluation.spec import ExperimentSpec, VariantSpec


def _episode(variant_id, seed, status=EpisodeStatus.COMPLETED, compat=CompatibilityStatus.SUPPORTED, mission_success=True, **metric_overrides):
    metrics = {
        "mission_success": mission_success,
        "mission_progress_ratio": 1.0 if mission_success else 0.5,
        "decisions_requested": 5,
        "first_pass_valid": 5,
        "repair_attempts": 0,
        "fallbacks": 0,
        "final_valid_llm_actions": 5,
    }
    metrics.update(metric_overrides)
    return EpisodeResult(
        experiment_id="exp1",
        variant_id=variant_id,
        episode_id=f"exp1__{variant_id}__seed{seed}",
        seed=seed,
        status=status,
        compatibility_status=compat,
        mission_success=mission_success if compat == CompatibilityStatus.SUPPORTED else None,
        game_time=90.0 if compat == CompatibilityStatus.SUPPORTED else None,
        wall_time=120.0 if compat == CompatibilityStatus.SUPPORTED else None,
        decision_count=5 if compat == CompatibilityStatus.SUPPORTED else 0,
        metrics=metrics if compat == CompatibilityStatus.SUPPORTED else {},
    )


def _variants():
    return (
        VariantSpec(variant_id="native_qwen", agent_runtime="native", provider="ollama", model="qwen:latest"),
        VariantSpec(variant_id="native_gemma4", agent_runtime="native", provider="ollama", model="gemma4:latest"),
        VariantSpec(variant_id="strands_gemma4", agent_runtime="strands", provider="ollama", model="gemma4:latest"),
    )


def _well_formed_result(seeds=(1, 2, 3)):
    spec = ExperimentSpec(experiment_id="exp1", variants=_variants(), seeds=seeds)
    episodes = [
        _episode(v.variant_id, seed)
        for v in _variants()
        for seed in seeds
    ]
    return ExperimentResult(
        experiment_id="exp1",
        spec=spec,
        episodes=episodes,
        config_snapshot={},
        started_at="t0",
        finished_at="t1",
    )


# ---------------------------------------------------------------------------
# Integrity checks (spec section 25, 32)
# ---------------------------------------------------------------------------


def test_integrity_passes_for_well_formed_result():
    result = _well_formed_result()
    report = verify_integrity(result)
    assert report.ok is True
    assert all(report.checks.values())
    assert report.problems == []


def test_integrity_detects_missing_episode():
    result = _well_formed_result()
    result.episodes.pop()  # drop one - now short of the full expansion
    report = verify_integrity(result)
    assert report.ok is False
    assert report.checks["no_missing_results"] is False


def test_integrity_detects_duplicate_episode_id():
    result = _well_formed_result()
    result.episodes.append(result.episodes[0])
    report = verify_integrity(result)
    assert report.ok is False
    assert report.checks["no_duplicate_episode_ids"] is False


def test_integrity_detects_seed_set_mismatch_across_variants():
    result = _well_formed_result()
    # Give native_gemma4 an extra seed native_qwen/strands_gemma4 don't have.
    result.episodes.append(_episode("native_gemma4", 999))
    report = verify_integrity(result)
    assert report.ok is False
    assert report.checks["same_seed_set_across_variants"] is False


def test_integrity_detects_aggregate_mismatch(monkeypatch):
    result = _well_formed_result()
    monkeypatch.setattr(result, "to_summary_dict", lambda: {"requested": 999, "supported": 0, "completed": 0, "successful": 0})
    report = verify_integrity(result)
    assert report.ok is False
    assert report.checks["aggregate_reconciles_with_raw_episodes"] is False


def test_integrity_decision_episode_correlation(tmp_path):
    result = _well_formed_result(seeds=(1,))
    known_id = result.episodes[0].episode_id
    decisions_path = tmp_path / "decisions.jsonl"
    with open(decisions_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"event_type": "llm_decision", "episode_id": known_id}) + "\n")
        f.write(json.dumps({"event_type": "llm_decision", "episode_id": "exp1__ghost_variant__seed42"}) + "\n")

    report = verify_integrity(result, decisions_jsonl_path=str(decisions_path))
    assert report.checks["decision_episode_correlation"] is False
    assert any("ghost_variant" in p for p in report.problems)


def test_integrity_decision_counts_reconcile(tmp_path):
    variant = VariantSpec(variant_id="native_qwen", agent_runtime="native", provider="ollama", model="qwen:latest")
    spec = ExperimentSpec(experiment_id="exp1", variants=(variant,), seeds=(1,))
    episode = _episode("native_qwen", 1)  # decision_count=5
    result = ExperimentResult(experiment_id="exp1", spec=spec, episodes=[episode], config_snapshot={}, started_at="t0", finished_at="t1")

    decisions_path = tmp_path / "decisions.jsonl"
    with open(decisions_path, "w", encoding="utf-8") as f:
        for _ in range(5):
            f.write(json.dumps({"event_type": "llm_decision", "episode_id": episode.episode_id}) + "\n")

    report = verify_integrity(result, decisions_jsonl_path=str(decisions_path))
    assert report.checks["decision_counts_reconcile"] is True


def test_integrity_detects_decision_count_mismatch(tmp_path):
    result = _well_formed_result(seeds=(1,))
    known_id = result.episodes[0].episode_id  # decision_count=5
    decisions_path = tmp_path / "decisions.jsonl"
    with open(decisions_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"event_type": "llm_decision", "episode_id": known_id}) + "\n")  # only 1, not 5

    report = verify_integrity(result, decisions_jsonl_path=str(decisions_path))
    assert report.checks["decision_counts_reconcile"] is False


# ---------------------------------------------------------------------------
# Incompatible exclusion from gameplay aggregates (spec section 4, 32)
# ---------------------------------------------------------------------------


def test_incompatible_variant_never_enters_gameplay_aggregates():
    spec = ExperimentSpec(
        experiment_id="exp1",
        variants=_variants() + (VariantSpec(variant_id="strands_qwen", agent_runtime="strands", provider="ollama", model="qwen:latest"),),
        seeds=(1, 2),
    )
    episodes = [_episode(v.variant_id, seed) for v in _variants() for seed in (1, 2)]
    episodes += [
        _episode("strands_qwen", seed, status=EpisodeStatus.INCOMPATIBLE, compat=CompatibilityStatus.MODEL_CAPABILITY_MISMATCH, mission_success=None)
        for seed in (1, 2)
    ]
    result = ExperimentResult(experiment_id="exp1", spec=spec, episodes=episodes, config_snapshot={}, started_at="t0", finished_at="t1")

    summary = result.to_summary_dict()
    strands_qwen_summary = summary["by_variant"]["strands_qwen"]
    assert strands_qwen_summary["requested"] == 2
    assert strands_qwen_summary["supported"] == 0
    assert strands_qwen_summary["completed"] == 0
    assert strands_qwen_summary["successful"] == 0
    assert strands_qwen_summary["compatibility_statuses"] == {"MODEL_CAPABILITY_MISMATCH": 2}
    # The other three variants' successful counts are unaffected.
    assert summary["by_variant"]["native_qwen"]["successful"] == 2


# ---------------------------------------------------------------------------
# Paired per-seed alignment (spec section 18, 32)
# ---------------------------------------------------------------------------


def test_paired_seed_rows_align_by_seed():
    result = _well_formed_result(seeds=(1, 2, 3))
    rows = paired_seed_rows(result, "native_qwen", "native_gemma4")

    assert [r["seed"] for r in rows] == [1, 2, 3]
    for row in rows:
        assert row["native_qwen.status"] == "COMPLETED"
        assert row["native_gemma4.status"] == "COMPLETED"
        assert row["native_qwen.decisions_requested"] == 5
        assert row["native_gemma4.decisions_requested"] == 5


def test_paired_seed_rows_only_include_shared_supported_seeds():
    result = _well_formed_result(seeds=(1, 2, 3))
    # native_gemma4 has an extra seed native_qwen never ran.
    result.episodes.append(_episode("native_gemma4", 4))
    rows = paired_seed_rows(result, "native_qwen", "native_gemma4")
    assert [r["seed"] for r in rows] == [1, 2, 3]  # seed 4 excluded - not shared


def test_paired_seed_rows_exclude_incompatible_variant():
    spec = ExperimentSpec(
        experiment_id="exp1",
        variants=_variants() + (VariantSpec(variant_id="strands_qwen", agent_runtime="strands", provider="ollama", model="qwen:latest"),),
        seeds=(1, 2),
    )
    episodes = [_episode(v.variant_id, seed) for v in _variants() for seed in (1, 2)]
    episodes += [
        _episode("strands_qwen", seed, status=EpisodeStatus.INCOMPATIBLE, compat=CompatibilityStatus.MODEL_CAPABILITY_MISMATCH, mission_success=None)
        for seed in (1, 2)
    ]
    result = ExperimentResult(experiment_id="exp1", spec=spec, episodes=episodes, config_snapshot={}, started_at="t0", finished_at="t1")

    rows = paired_seed_rows(result, "native_gemma4", "strands_qwen")
    assert rows == []  # strands_qwen is never SUPPORTED, so no shared seed can pair


def test_paired_markdown_table_renders_for_no_shared_seeds():
    result = _well_formed_result(seeds=(1,))
    table = render_paired_markdown_table(result, "native_qwen", "does_not_exist")
    assert "No shared supported seeds" in table


def test_paired_markdown_table_renders_rows():
    result = _well_formed_result(seeds=(1, 2))
    table = render_paired_markdown_table(result, "native_gemma4", "strands_gemma4")
    assert "native_gemma4.status" in table
    assert "strands_gemma4.status" in table
    assert table.count("\n") >= 3  # header + separator + at least one data row
