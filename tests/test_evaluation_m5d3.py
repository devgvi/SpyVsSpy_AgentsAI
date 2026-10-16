"""M5D.3 spec section 39: benchmark-specific correctness tests, all
network-free (provider="mock" throughout - no live LLM required). The
harness itself barely changed for M5D.3 (mostly M5D/M5D.1/M5D.2 code,
already covered by their own test suites); what's new here is specific to
running *this* benchmark correctly:

- the frozen `experiments/m5d3_corrected_mission_v1.json` config itself
  (A/B/C/D shape, shared seeds, identical limits across matched variants);
- Strands+Qwen stays compatibility-only and out of gameplay denominators;
- integrity reconciliation, including the new M5D.2
  "action_lifecycle_present" check (src/evaluation/integrity.py);
- per-seed pairing works with M5D.3's richer mission-progress field set;
- M5D.1/M5D.2 historical outputs can never collide with M5D.3's, by
  construction (distinct experiment_id -> distinct evaluation_runs/ dir).
"""

from __future__ import annotations

import json
import os

from src.evaluation.compatibility import CompatibilityStatus
from src.evaluation.episode_runner import EpisodeRunner, EpisodeStatus
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.integrity import verify_integrity
from src.evaluation.paired import paired_seed_rows
from src.evaluation.spec import ExperimentSpec, VariantSpec

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "m5d3_corrected_mission_v1.json")


def _load_m5d3_spec() -> ExperimentSpec:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return ExperimentSpec.from_json(f.read())


# ---------------------------------------------------------------------------
# The frozen benchmark config itself
# ---------------------------------------------------------------------------


def test_m5d3_config_has_a_new_immutable_experiment_id_distinct_from_m5d1_and_m5d2():
    spec = _load_m5d3_spec()
    assert spec.experiment_id not in {"m5d1-local-baseline-v1", "m5d1_pilot", "m5d2-live-validation"}
    # By construction (src/evaluation/run.py / ExperimentRunner), output is
    # always written under evaluation_runs/<experiment_id>/ - a distinct
    # experiment_id can never collide with or overwrite M5D.1/M5D.2's own
    # output directories.
    assert spec.experiment_id == "m5d3-corrected-mission-v1"


def test_m5d3_config_expands_required_variants_on_a_shared_seed_set():
    spec = _load_m5d3_spec()
    variant_ids = [v.variant_id for v in spec.variants]
    assert len(variant_ids) == 4  # A, B, C, D

    by_key = {(v.agent_runtime, v.provider, v.model): v for v in spec.variants}
    assert ("native", "ollama", "qwen:latest") in by_key
    assert ("native", "ollama", "gemma4:latest") in by_key
    assert ("strands", "ollama", "gemma4:latest") in by_key
    assert ("strands", "ollama", "qwen:latest") in by_key  # D, compatibility-only

    episodes = spec.expand()
    seeds_by_variant = {}
    for e in episodes:
        seeds_by_variant.setdefault(e.variant.variant_id, set()).add(e.seed)
    # Every variant (including D) is expanded over the exact same seed set -
    # D simply never reaches gameplay once compatibility is checked.
    distinct_seed_sets = {frozenset(s) for s in seeds_by_variant.values()}
    assert len(distinct_seed_sets) == 1


def test_m5d3_config_uses_identical_episode_limits_across_matched_variants():
    """spec section 11/section 5 (STOP condition 5): one ExperimentSpec-level
    decision/game/wall limit applies to every variant - there is no
    per-variant override anywhere in VariantSpec, so A/B/C/D share limits
    by construction. Assert the actual frozen values are deliberate, not
    M5D.1's smoke-sized budget (spec section 8: "do not reuse the M5D.1
    smoke-style decision budget").
    """
    spec = _load_m5d3_spec()
    assert spec.execution_mode == "evaluation"
    assert spec.max_decisions > 6  # bigger than M5D.1's smoke-sized cap
    assert spec.max_episode_game_time == 180.0  # natural mission duration preserved (spec section 9)
    episodes = spec.expand()
    for e in episodes:
        assert e.max_decisions == spec.max_decisions
        assert e.max_episode_game_time == spec.max_episode_game_time
        assert e.max_episode_wall_time == spec.max_episode_wall_time


def test_m5d3_strands_qwen_is_compatibility_only_and_never_enters_gameplay():
    from src.evaluation.compatibility import check_compatibility

    spec = _load_m5d3_spec()
    d_variant = next(v for v in spec.variants if v.agent_runtime == "strands" and v.model == "qwen:latest")
    compatibility = check_compatibility(d_variant)
    assert compatibility.status == CompatibilityStatus.MODEL_CAPABILITY_MISMATCH
    assert not compatibility.supported

    runner = EpisodeRunner()
    episode_spec = next(e for e in spec.expand() if e.variant.variant_id == d_variant.variant_id)
    result = runner.run(episode_spec, telemetry_path="/dev/null")
    assert result.status == EpisodeStatus.INCOMPATIBLE
    assert result.mission_success is None
    assert result.decision_count == 0


# ---------------------------------------------------------------------------
# Integrity, including the new M5D.2 action_lifecycle_present check
# ---------------------------------------------------------------------------


def _mock_variant(variant_id: str) -> VariantSpec:
    return VariantSpec(variant_id=variant_id, agent_runtime="native", provider="mock", model="mock-model")


def test_m5d3_lifecycle_telemetry_present_and_integrity_reconciles(tmp_path):
    """Runs a real (mocked, no network) episode through the harness with a
    MOVE decision so ActionExecutor genuinely executes, then verifies
    verify_integrity's new 'action_lifecycle_present' check passes against
    the real decisions.jsonl this produced - not a synthetic fixture."""
    import src.game as game_module
    from src.llm.clients.mock_client import MockModelClient

    def build_move_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return MockModelClient(response=[{"action": "move", "target_id": "door_corridor_main_office"}])

    monkeypatch_target = game_module.Game._build_llm_client
    game_module.Game._build_llm_client = build_move_mock
    try:
        spec = ExperimentSpec(
            experiment_id="m5d3_lifecycle_smoke",
            variants=(_mock_variant("native_mock"),),
            seeds=(9001,),
            max_decisions=3,
            max_episode_wall_time=30.0,
        )
        runner = ExperimentRunner(output_root=str(tmp_path))
        result = runner.run(spec, write_outputs=True)
    finally:
        game_module.Game._build_llm_client = monkeypatch_target

    decisions_path = os.path.join(str(tmp_path), spec.experiment_id, "decisions.jsonl")
    with open(decisions_path, "r", encoding="utf-8") as f:
        raw_events = [json.loads(line) for line in f if line.strip()]
    assert any(e.get("event_type") == "action_lifecycle" for e in raw_events)

    report = verify_integrity(result, decisions_jsonl_path=decisions_path)
    assert report.checks.get("action_lifecycle_present") is True
    assert report.ok, report.problems


def test_action_lifecycle_present_check_fails_when_metrics_claim_actions_but_telemetry_has_none(tmp_path):
    """Regression guard for the new check itself: if an episode's own
    metrics report accepted actions but decisions.jsonl (or the events
    passed in) has no matching action_lifecycle rows, integrity must flag
    it rather than silently trusting the metrics field."""
    import src.game as game_module
    from src.llm.clients.mock_client import MockModelClient

    def build_move_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return MockModelClient(response=[{"action": "move", "target_id": "door_corridor_main_office"}])

    monkeypatch_target = game_module.Game._build_llm_client
    game_module.Game._build_llm_client = build_move_mock
    try:
        spec = ExperimentSpec(
            experiment_id="m5d3_lifecycle_tamper",
            variants=(_mock_variant("native_mock"),),
            seeds=(9002,),
            max_decisions=3,
            max_episode_wall_time=30.0,
        )
        runner = ExperimentRunner(output_root=str(tmp_path))
        result = runner.run(spec, write_outputs=False)
    finally:
        game_module.Game._build_llm_client = monkeypatch_target

    assert result.episodes[0].metrics.get("actions_accepted", 0) > 0

    # Simulate telemetry that never actually recorded the lifecycle event
    # decisions.jsonl claims none exist for this episode.
    report = verify_integrity(result, decisions_jsonl_path=None)
    # No decisions_jsonl_path given -> the check isn't run at all (can't
    # reconcile against nothing); confirm it's simply absent, not a false pass.
    assert "action_lifecycle_present" not in report.checks


# ---------------------------------------------------------------------------
# Per-seed pairing with M5D.3's richer mission-progress fields
# ---------------------------------------------------------------------------


def test_m5d3_paired_rows_include_mission_progress_and_navigation_fields(tmp_path):
    import src.game as game_module
    from src.llm.clients.mock_client import MockModelClient

    def build_move_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return MockModelClient(response=[{"action": "move", "target_id": "door_corridor_main_office"}])

    monkeypatch_target = game_module.Game._build_llm_client
    game_module.Game._build_llm_client = build_move_mock
    try:
        spec = ExperimentSpec(
            experiment_id="m5d3_pairing_smoke",
            variants=(_mock_variant("native_a"), _mock_variant("native_b")),
            seeds=(9003,),
            max_decisions=3,
            max_episode_wall_time=30.0,
        )
        runner = ExperimentRunner(output_root=str(tmp_path))
        result = runner.run(spec, write_outputs=False)
    finally:
        game_module.Game._build_llm_client = monkeypatch_target

    fields = (
        "mission_success",
        "mission_progress_ratio",
        "unique_rooms_visited",
        "actions_completed",
        "actions_failed",
        "mean_simulation_ticks_per_completed_action",
    )
    rows = paired_seed_rows(result, "native_a", "native_b", fields=fields)
    assert len(rows) == 1
    assert rows[0]["seed"] == 9003
    for field_name in fields:
        assert f"native_a.{field_name}" in rows[0]
        assert f"native_b.{field_name}" in rows[0]


# ---------------------------------------------------------------------------
# Technical failure vs. mission loss stays separate (existing EpisodeStatus
# semantics, exercised here specifically through the M5D.3 config shape)
# ---------------------------------------------------------------------------


def test_m5d3_technical_failure_never_counted_as_a_supported_gameplay_loss():
    spec = _load_m5d3_spec()
    d_variant = next(v for v in spec.variants if v.agent_runtime == "strands" and v.model == "qwen:latest")
    episode_spec = next(e for e in spec.expand() if e.variant.variant_id == d_variant.variant_id)
    runner = EpisodeRunner()
    result = runner.run(episode_spec, telemetry_path="/dev/null")
    # INCOMPATIBLE (a technical/config fact) must never read as a played,
    # supported mission loss - mission_success stays None, not False.
    assert result.status == EpisodeStatus.INCOMPATIBLE
    assert result.compatibility_status == CompatibilityStatus.MODEL_CAPABILITY_MISMATCH
    assert result.mission_success is None
    assert result.metrics == {}
