"""M5D.7 spec section 31: deterministic, network-free tests for the
controlled runtime-representation experiment infrastructure. The live
scenario runs themselves (real Ollama/Gemma4) are not exercised here -
see CHANGELOG.md's M5D.7 entry for those results - only the machinery
that must be correct before any live call is trusted.
"""

from __future__ import annotations

import re

from src.actions import ActionType
from src.agents.action_lifecycle import ActionExecutor
from src.evaluation.representation_experiment import (
    build_scenarios,
    render_summary_markdown,
    summarize,
    verify_scenario_legality,
)
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import (
    DEFAULT_SYSTEM_PROMPT,
    NATIVE_CURRENT_REPRESENTATION,
    NATIVE_TOOL_LIKE_REPRESENTATION,
    NATIVE_TOOL_LIKE_SYSTEM_PROMPT,
    STRANDS_TOOL_REPRESENTATION,
    LLMController,
    LLMControllerConfig,
    compact_observation_for_model,
    convert_response_to_action,
    enumerate_legal_actions,
    format_legal_actions_tool_like,
    serialize_observation,
)


def _parse_tool_like_text(text: str) -> set[tuple[str, str | None]]:
    """Reconstructs the semantic (action, target_id) set from Variant B's
    formatted text block, purely for the parity test below - never used
    by production code (the model itself just reads the text). M5E.4.2:
    a target line may carry a trailing `[...]` informational annotation
    (door state/effect) - stripped here since it's display detail, not a
    change to which targets are legal (target_ids never contain spaces or
    `[`, so matching up to the first whitespace/bracket is unambiguous)."""
    pairs: set[tuple[str, str | None]] = set()
    current_action: str | None = None
    for line in text.splitlines():
        header = re.match(r"^([A-Z_]+)\(target_id\)$", line)
        if header:
            current_action = header.group(1).lower()
            continue
        target_line = re.match(r"^- (\S+)", line)
        if target_line and current_action:
            pairs.add((current_action, target_line.group(1)))
        if line == "No target required." and current_action:
            pairs.add((current_action, None))
    return pairs


# ---------------------------------------------------------------------------
# Scenario fixtures self-check
# ---------------------------------------------------------------------------


def test_all_scenarios_pass_their_own_deterministic_legality_check():
    for scenario in build_scenarios():
        verify_scenario_legality(scenario)  # raises on mismatch


def test_scenario_legality_check_actually_detects_a_mismatch():
    from dataclasses import replace

    scenario = build_scenarios()[0]
    tampered = replace(scenario, expected_legal=frozenset({("inspect", "desk_01")}))  # false claim: far desk inspectable
    try:
        verify_scenario_legality(tampered)
        assert False, "expected AssertionError for a deliberately wrong legality claim"
    except AssertionError:
        pass


# ---------------------------------------------------------------------------
# A/B semantic legal-action parity (spec section 14)
# ---------------------------------------------------------------------------


def test_variant_a_and_b_present_semantically_identical_legal_actions():
    for scenario in build_scenarios():
        observation_payload = serialize_observation(scenario.observation)
        legal_set = {(e["action"], e.get("target_id")) for e in observation_payload["legal_actions"]}

        variant_a = compact_observation_for_model(observation_payload, NATIVE_CURRENT_REPRESENTATION)
        a_set = {(action, target) for action, targets in variant_a["legal_actions_compact"].items() for target in (targets or [None])}

        variant_b = compact_observation_for_model(observation_payload, NATIVE_TOOL_LIKE_REPRESENTATION)
        b_set = _parse_tool_like_text(variant_b["available_actions_text"])

        assert a_set == legal_set, scenario.scenario_id
        assert b_set == legal_set, scenario.scenario_id


def test_variant_b_available_actions_text_matches_the_direct_formatter():
    scenario = build_scenarios()[1]  # near_object
    observation_payload = serialize_observation(scenario.observation)
    variant_b = compact_observation_for_model(observation_payload, NATIVE_TOOL_LIKE_REPRESENTATION)
    assert variant_b["available_actions_text"] == format_legal_actions_tool_like(observation_payload["legal_actions"])


def test_variant_a_never_gets_available_actions_text():
    scenario = build_scenarios()[0]
    observation_payload = serialize_observation(scenario.observation)
    variant_a = compact_observation_for_model(observation_payload, NATIVE_CURRENT_REPRESENTATION)
    assert "available_actions_text" not in variant_a


# ---------------------------------------------------------------------------
# near_object_id parity across variants
# ---------------------------------------------------------------------------


def test_near_object_id_identical_across_variants():
    for scenario in build_scenarios():
        observation_payload = serialize_observation(scenario.observation)
        variant_a = compact_observation_for_model(observation_payload, NATIVE_CURRENT_REPRESENTATION)
        variant_b = compact_observation_for_model(observation_payload, NATIVE_TOOL_LIKE_REPRESENTATION)
        assert variant_a["self"]["near_object_id"] == variant_b["self"]["near_object_id"] == scenario.observation.nearby_object_id


# ---------------------------------------------------------------------------
# B does not invoke real tool calling / stays on Native path
# ---------------------------------------------------------------------------


def test_variant_b_is_ordinary_native_controller_no_strands():
    scenario = build_scenarios()[1]
    telemetry = TelemetryLogger()
    config = LLMControllerConfig(system_prompt=NATIVE_TOOL_LIKE_SYSTEM_PROMPT, representation_variant=NATIVE_TOOL_LIKE_REPRESENTATION)
    controller = LLMController(
        "white",
        MockModelClient({"action": "inspect", "target_id": "statue_01"}),
        config=config,
        telemetry=telemetry,
    )
    assert isinstance(controller, LLMController)  # the exact same class as Variant A, not a Strands adapter
    controller.get_action(scenario.observation)
    controller._thinking_thread.join(timeout=2.0)

    event = telemetry.events[-1]
    assert event["representation_variant"] == NATIVE_TOOL_LIKE_REPRESENTATION
    assert event["agent_runtime"] == "native"
    # Native never has runtime_metadata (that's Strands-only, M5C.1) -
    # confirms this decision took the plain Native path, not a tool call.
    assert event.get("runtime_metadata") is None


def test_variant_b_uses_same_strict_validator_and_repair_v2():
    """Variant B's proposals are validated/repaired through the identical
    shared functions Variant A uses - no second validation implementation."""
    from src.llm_controller import REPAIR_PROMPT_VERSION

    scenario = build_scenarios()[0]  # far_object: illegal first attempt
    telemetry = TelemetryLogger()
    config = LLMControllerConfig(system_prompt=NATIVE_TOOL_LIKE_SYSTEM_PROMPT, representation_variant=NATIVE_TOOL_LIKE_REPRESENTATION)
    controller = LLMController(
        "white",
        MockModelClient([{"action": "inspect", "target_id": "desk_01"}, {"action": "move", "target_id": "desk_01"}]),
        config=config,
        telemetry=telemetry,
    )
    controller.get_action(scenario.observation)
    controller._thinking_thread.join(timeout=2.0)

    event = telemetry.events[-1]
    assert event["initial_error_type"] == "out_of_range"  # same categories as Variant A (M5D.5)
    assert event["repair_context_version"] == REPAIR_PROMPT_VERSION  # same repair-v2
    assert event["recovery_outcome"] == "REPAIRED_VALID"


# ---------------------------------------------------------------------------
# C normalizes to the same canonical GameAction contract
# ---------------------------------------------------------------------------


def test_all_variants_normalize_the_same_proposal_to_the_same_game_action():
    scenario = build_scenarios()[1]  # near_object: statue_01
    legal_actions = enumerate_legal_actions(scenario.observation)
    proposal = {"action": "inspect", "target_id": "statue_01"}

    action = convert_response_to_action(scenario.observation, proposal, legal_actions)
    assert action is not None
    assert action.type == ActionType.INSPECT
    assert action.target_id == "statue_01"

    # ActionExecutor (shared by every runtime/variant, M5D.2) accepts the
    # identical normalized action/target the same way regardless of which
    # representation variant produced the proposal.
    executor = ActionExecutor()
    game_action, boundary, _action_id = executor.start(scenario.observation, "dec-1", "inspect", "statue_01", None, 90.0)
    assert game_action.type == ActionType.INSPECT
    assert boundary == "resolved"


# ---------------------------------------------------------------------------
# Representation-version telemetry
# ---------------------------------------------------------------------------


def test_representation_variant_recorded_in_telemetry_for_both_native_variants():
    scenario = build_scenarios()[1]
    for variant, prompt in (
        (NATIVE_CURRENT_REPRESENTATION, DEFAULT_SYSTEM_PROMPT),
        (NATIVE_TOOL_LIKE_REPRESENTATION, NATIVE_TOOL_LIKE_SYSTEM_PROMPT),
    ):
        telemetry = TelemetryLogger()
        config = LLMControllerConfig(system_prompt=prompt, representation_variant=variant)
        controller = LLMController("white", MockModelClient({"action": "inspect", "target_id": "statue_01"}), config=config, telemetry=telemetry)
        controller.get_action(scenario.observation)
        controller._thinking_thread.join(timeout=2.0)
        assert telemetry.events[-1]["representation_variant"] == variant


# ---------------------------------------------------------------------------
# Aggregation grouped by representation variant
# ---------------------------------------------------------------------------


def test_summarize_groups_by_representation_variant_and_scenario():
    rows = [
        {"scenario_id": "far_object", "representation_variant": "native_current-v1", "initial_model_output_valid": True, "repair_attempted": False, "repair_valid": None, "final_model_action_valid": True, "latency_ms": 100.0, "input_tokens": 50, "output_tokens": 10, "initial_error_type": None},
        {"scenario_id": "far_object", "representation_variant": "native_current-v1", "initial_model_output_valid": False, "repair_attempted": True, "repair_valid": True, "final_model_action_valid": True, "latency_ms": 200.0, "input_tokens": 60, "output_tokens": 20, "initial_error_type": "out_of_range"},
        {"scenario_id": "near_object", "representation_variant": "native_tool_like-v1", "initial_model_output_valid": True, "repair_attempted": False, "repair_valid": None, "final_model_action_valid": True, "latency_ms": 150.0, "input_tokens": 55, "output_tokens": 15, "initial_error_type": None},
    ]
    summary = summarize(rows)

    assert set(summary["by_variant"].keys()) == {"native_current-v1", "native_tool_like-v1"}
    a = summary["by_variant"]["native_current-v1"]
    assert a["n"] == 2
    assert a["first_pass_valid"] == 1
    assert a["first_pass_valid_rate"] == 0.5
    assert a["repair_attempts"] == 1
    assert a["repair_successes"] == 1
    assert a["repair_success_rate"] == 1.0
    assert a["error_type_breakdown"] == {"out_of_range": 1}

    assert "far_object" in summary["by_variant_by_scenario"]["native_current-v1"]
    assert "near_object" in summary["by_variant_by_scenario"]["native_tool_like-v1"]

    markdown = render_summary_markdown(summary)
    assert "native_current-v1" in markdown
    assert "native_tool_like-v1" in markdown
