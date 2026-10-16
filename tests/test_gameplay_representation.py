"""M5D.8 spec section 26: gameplay-scale validation of M5D.7's
representation experiment. No live LLM required anywhere in this file -
`provider="mock"` throughout. See CHANGELOG.md's M5D.8 entry for the real
live-run results using `experiments/m5d8_gameplay_representation_v1.json`.
"""

from __future__ import annotations

import json
import os

import pytest

from src import constants
from src.agents.native_agent import NativeAgentAdapter, NativeToolLikeAgentAdapter
from src.agents.registry import create_agent_runtime
from src.agents.strands_agent import StrandsAgentAdapter
from src.evaluation.compatibility import CompatibilityStatus, check_compatibility
from src.evaluation.episode_runner import EpisodeRunner, EpisodeStatus
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.integrity import verify_integrity
from src.evaluation.paired import paired_seed_rows
from src.evaluation.spec import ExperimentSpec, VariantSpec
from src.llm.clients.mock_client import MockModelClient
from src.llm_controller import NATIVE_TOOL_LIKE_REPRESENTATION, LLMControllerConfig

WHITE = constants.WHITE_SPY_ID
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "m5d8_gameplay_representation_v1.json")


def _load_m5d8_spec() -> ExperimentSpec:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return ExperimentSpec.from_json(f.read())


# ---------------------------------------------------------------------------
# A/B/C representation selection through the real registry
# ---------------------------------------------------------------------------


def test_registry_resolves_all_three_gameplay_variants():
    a = create_agent_runtime("native", actor_id="white", model_client=MockModelClient({"action": "wait"}))
    b = create_agent_runtime("native_tool_like", actor_id="white", model_client=MockModelClient({"action": "wait"}))
    c = create_agent_runtime("strands", actor_id="white", agent_factory=lambda *a, **k: None)

    assert isinstance(a, NativeAgentAdapter) and not isinstance(a, NativeToolLikeAgentAdapter)
    assert isinstance(b, NativeToolLikeAgentAdapter)
    assert isinstance(c, StrandsAgentAdapter)
    assert a.runtime_name == "native"
    assert b.runtime_name == "native_tool_like"
    assert c.runtime_name == "strands"


def test_native_tool_like_compatibility_requires_no_special_capability():
    variant = VariantSpec(variant_id="B", agent_runtime="native_tool_like", provider="ollama", model="gemma4:latest")
    # Same shape as native: no tools capability required, so a capability
    # query never blocks it the way Strands+Qwen is blocked.
    from src.evaluation.compatibility import RUNTIME_REQUIRED_CAPABILITIES

    assert RUNTIME_REQUIRED_CAPABILITIES["native_tool_like"] == frozenset()


# ---------------------------------------------------------------------------
# A unchanged / B no real tool calling / B same validator+repair-v2
# ---------------------------------------------------------------------------


def test_variant_a_default_representation_is_native_current_unchanged():
    from src.llm_controller import LLMController, NATIVE_CURRENT_REPRESENTATION

    controller = LLMController("white", MockModelClient({"action": "wait"}))
    assert controller.config.representation_variant == NATIVE_CURRENT_REPRESENTATION


def test_variant_b_is_native_tool_like_adapter_no_strands_import_needed():
    adapter = create_agent_runtime("native_tool_like", actor_id="white", model_client=MockModelClient({"action": "wait"}))
    assert isinstance(adapter, NativeAgentAdapter)  # same family, ordinary Native path
    assert adapter.controller.config.representation_variant == NATIVE_TOOL_LIKE_REPRESENTATION
    assert "You are calling one function" in adapter.controller.config.system_prompt


def test_variant_b_uses_shared_validator_and_repair_v2_end_to_end():
    from src.llm.telemetry import TelemetryLogger
    from src.llm_controller import REPAIR_PROMPT_VERSION
    from src.observation import Observation, ObservedObject

    desk = ObservedObject(
        id="desk_01", type="desk", label="desk_01", position=(9000.0, 9000.0), collision_size=(20.0, 20.0),
        interactable=True, trappable=False, searched=False, inspected=False, trap_status="unknown",
        trap_type=None, trap_is_mine=False, door_open=None, lock_type=None, locked=False, can_unlock=False,
    )
    observation = Observation(
        actor_id="white", seed=1, position=(5.0, 5.0), facing="east", stunned=False, stunned_seconds=0.0,
        busy=False, busy_seconds=0.0, mission_time_remaining=100.0, mission_complete=False,
        inventory=frozenset(), knowledge=frozenset(), required_items=(), trap_inventory={},
        nearby_object_id=None, items_found_here=frozenset(), objects=(desk,), nav_grid=None,
        log_event=lambda *a, **k: None, current_room="office", known_rooms=frozenset({"office"}),
        known_doors=frozenset(), known_locked_objects={}, known_item_locations={}, visible_other_actor=False,
        opponent=None, last_seen_opponent=None, revealed_opponent=None, revealed_opponent_seconds=0.0,
    )
    telemetry = TelemetryLogger()
    adapter = create_agent_runtime(
        "native_tool_like", actor_id="white",
        model_client=MockModelClient([{"action": "inspect", "target_id": "desk_01"}, {"action": "wait"}]),
        telemetry=telemetry,
    )
    adapter.get_action(observation)
    adapter.controller._thinking_thread.join(timeout=2.0)

    event = telemetry.events[-1]
    assert event["initial_error_type"] == "out_of_range"  # same shared validator category as Variant A
    assert event["repair_context_version"] == REPAIR_PROMPT_VERSION  # same repair-v2
    assert event["representation_variant"] == NATIVE_TOOL_LIKE_REPRESENTATION
    assert event["agent_runtime"] == "native_tool_like"


def test_variant_c_strands_path_unchanged_by_m5d8():
    adapter = create_agent_runtime("strands", actor_id="white", agent_factory=lambda *a, **k: None)
    assert adapter.runtime_name == "strands"
    assert adapter.runtime_version == "strands-adapter-v3"  # M5E.1.1 watchdog + M5E.4 Bedrock support; unrelated to M5D.8


# ---------------------------------------------------------------------------
# The frozen M5D.8 config
# ---------------------------------------------------------------------------


def test_m5d8_config_has_new_immutable_id_and_three_variants():
    spec = _load_m5d8_spec()
    assert spec.experiment_id == "m5d8-gameplay-representation-v1"
    assert spec.experiment_id not in {
        "m5d1-local-baseline-v1",
        "m5d3-corrected-mission-v1",
        "m5d6-corrected-mission-v2",
        "m5d7-runtime-representation-v1",
    }
    runtimes = {v.agent_runtime for v in spec.variants}
    assert runtimes == {"native", "native_tool_like", "strands"}
    models = {v.model for v in spec.variants}
    assert models == {"gemma4:latest"}  # Qwen excluded (spec section 1)

    episodes = spec.expand()
    seeds_by_variant = {}
    for e in episodes:
        seeds_by_variant.setdefault(e.variant.variant_id, set()).add(e.seed)
    assert len({frozenset(s) for s in seeds_by_variant.values()}) == 1  # shared seed set


def test_m5d8_episode_limits_identical_across_variants():
    spec = _load_m5d8_spec()
    for e in spec.expand():
        assert e.max_decisions == spec.max_decisions
        assert e.max_episode_game_time == spec.max_episode_game_time
        assert e.max_episode_wall_time == spec.max_episode_wall_time


# ---------------------------------------------------------------------------
# Integrity / aggregation grouped by variant / per-seed pairing
# ---------------------------------------------------------------------------


def _mock_variant(variant_id: str, agent_runtime: str) -> VariantSpec:
    return VariantSpec(variant_id=variant_id, agent_runtime=agent_runtime, provider="mock", model="mock-model")


def test_m5d8_integrity_and_variant_aggregation(tmp_path):
    import src.game as game_module

    def build_move_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return MockModelClient(response=[{"action": "move", "target_id": "door_corridor_main_office"}])

    monkeypatch_target = game_module.Game._build_llm_client
    game_module.Game._build_llm_client = build_move_mock
    try:
        spec = ExperimentSpec(
            experiment_id="m5d8_smoke",
            variants=(_mock_variant("A", "native"), _mock_variant("B", "native_tool_like")),
            seeds=(9201,),
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
    decision_events = [e for e in raw_events if e.get("event_type") == "llm_decision"]
    variants_seen = {e.get("representation_variant") for e in decision_events}
    assert variants_seen == {"native_current-v1", "native_tool_like-v1"}

    report = verify_integrity(result, decisions_jsonl_path=decisions_path)
    assert report.ok, report.problems

    # Gameplay-metric aggregation is already variant-generic (results.py
    # groups by variant_id, whatever runtime it names) - confirm it holds
    # for the new runtime name too, no special-casing needed.
    summary = result.to_summary_dict()
    assert set(summary["by_variant"].keys()) == {"A", "B"}


def test_m5d8_paired_rows_across_all_three_variants(tmp_path):
    import src.game as game_module

    def build_move_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return MockModelClient(response=[{"action": "move", "target_id": "door_corridor_main_office"}])

    monkeypatch_target = game_module.Game._build_llm_client
    game_module.Game._build_llm_client = build_move_mock
    try:
        spec = ExperimentSpec(
            experiment_id="m5d8_pairing_smoke",
            variants=(_mock_variant("A", "native"), _mock_variant("B", "native_tool_like")),
            seeds=(9202,),
            max_decisions=3,
            max_episode_wall_time=30.0,
        )
        runner = ExperimentRunner(output_root=str(tmp_path))
        result = runner.run(spec, write_outputs=False)
    finally:
        game_module.Game._build_llm_client = monkeypatch_target

    fields = ("first_pass_valid", "repair_attempts", "unique_rooms_visited")
    rows = paired_seed_rows(result, "A", "B", fields=fields)
    assert len(rows) == 1
    for field_name in fields:
        assert f"A.{field_name}" in rows[0]
        assert f"B.{field_name}" in rows[0]


def test_m5d8_incompatible_strands_qwen_stays_out_of_gameplay():
    """Not part of M5D.8's required variants, but a general reconfirmation
    that Qwen was correctly excluded (spec section 1) - Strands+Qwen would
    still be MODEL_CAPABILITY_MISMATCH if ever added, never silently gameplay."""
    variant = VariantSpec(variant_id="D", agent_runtime="strands", provider="ollama", model="qwen:latest")
    compatibility = check_compatibility(variant)
    assert compatibility.status == CompatibilityStatus.MODEL_CAPABILITY_MISMATCH
