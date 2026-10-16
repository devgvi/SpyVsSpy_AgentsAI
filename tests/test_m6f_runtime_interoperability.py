"""M6F spec section 24 (Phase B): network-free equivalence tests for the
new canonical AgentRequest/AgentResponse boundary and RP13-RP18. No live
model calls anywhere in this file - all Strands-side assertions use
`StrandsAgentAdapter._make_tool` directly (a pure closure, no real SDK
network path) or dead-reckon from `strands_model_payload`, which is
provably the same function as `native_model_payload` (HP02 - no sibling
reimplementation to drift).
"""

from __future__ import annotations

import copy

from src.agents.strands_agent import StrandsAgentAdapter
from src.evaluation.agent_contract import (
    FIXED_TEMPERATURE,
    build_agent_request,
    build_agent_response,
    native_model_payload,
    native_system_prompt,
    normalize_raw_output,
    strands_model_payload,
)
from src.evaluation.m6f_scenarios import (
    S0_ORDINARY_DECISION,
    S1_MEMORY_DECISION,
    S2_REPAIR_DECISION,
    S3_TOOL_ACTION_DECISION,
    build_s0_ordinary_decision,
    build_s1_memory_decision,
    build_s2_repair_decisions,
    build_s3_tool_action_decision,
)
from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST
from src.llm_controller import VALID_ACTION_NAMES, _normalize_action_name, format_legal_actions_tool_like, place_trap_repair_hints

MODEL_ID = "test-model-id"
PROVIDER = "bedrock"


# ---------------------------------------------------------------------------
# STOP CONDITION items 6-7: RP13-RP18 exist, manifest has no unexplained FAIL
# ---------------------------------------------------------------------------


def test_rp13_through_rp18_are_present_and_never_renumber_rp01_rp12():
    ids = [e.entry_id for e in RUNTIME_PARITY_MANIFEST]
    assert ids[:12] == [
        "RP01_PERSONALITY", "RP02_MODEL_CONFIG", "RP03_ACTION_CONTRACT", "RP04_VALIDATION", "RP05_REPAIR_SEMANTICS",
        "RP06_LEGAL_TRAP_HINTS", "RP07_FALLBACK", "RP08_FAILURE_COUNTERS", "RP09_COMMUNICATION_FLAG",
        "RP10_MEMORY_FLAGS", "RP11_TELEMETRY", "RP12_LEGAL_TARGET_HINTS",
    ]
    assert ids[12:] == [
        "RP13_AGENT_REQUEST_FIELDS", "RP14_GENERATION_SETTINGS", "RP15_REPAIR_CONTEXT",
        "RP16_TOOL_SCHEMA_SEMANTICS", "RP17_OUTPUT_NORMALIZATION", "RP18_PROVIDER_METADATA",
    ]
    assert len(RUNTIME_PARITY_MANIFEST) == 18


def test_manifest_has_no_unexplained_fail_before_live_comparison():
    assert all(e.status != FAIL for e in RUNTIME_PARITY_MANIFEST)


# ---------------------------------------------------------------------------
# RP13 - AgentRequest fields consumed equivalently by both runtimes
# ---------------------------------------------------------------------------


def test_rp13_native_and_strands_payloads_carry_identical_agent_request_fields():
    scenario = build_s0_ordinary_decision(MODEL_ID, PROVIDER)
    native_payload = native_model_payload(scenario.request)
    strands_payload = strands_model_payload(scenario.request)
    assert native_payload == strands_payload
    for key in ("self", "room", "memory", "opponent", "mission", "legal_actions_compact", "strategy_memory"):
        assert key in native_payload


def test_rp13_agent_request_semantic_hash_is_deterministic():
    a = build_s0_ordinary_decision(MODEL_ID, PROVIDER).request
    b = build_s0_ordinary_decision(MODEL_ID, PROVIDER).request
    assert a.request_semantic_hash() == b.request_semantic_hash()
    assert a.observation_hash() == b.observation_hash()
    assert a.memory_hash() == b.memory_hash()
    assert a.legal_actions_hash() == b.legal_actions_hash()


# ---------------------------------------------------------------------------
# RP14 - generation settings
# ---------------------------------------------------------------------------


def test_rp14_generation_settings_hash_stable_and_temperature_fixed():
    scenario = build_s0_ordinary_decision(MODEL_ID, PROVIDER)
    assert scenario.request.generation_settings["temperature"] == FIXED_TEMPERATURE == 0.0
    assert scenario.request.generation_settings["model_id"] == MODEL_ID
    again = build_s0_ordinary_decision(MODEL_ID, PROVIDER)
    assert scenario.request.generation_settings_hash() == again.request.generation_settings_hash()


# ---------------------------------------------------------------------------
# RP15 - repair context (RepairFacts) reaches both runtimes identically
# ---------------------------------------------------------------------------


def test_rp15_repair_context_hash_identical_across_runtimes():
    repairs = build_s2_repair_decisions(MODEL_ID, PROVIDER)
    assert len(repairs) == 4  # 2 failure families x 2 diagnostic contracts (spec sections 37, 48)
    for rs in repairs:
        assert rs.request.repair_context is not None
        assert rs.request.repair_facts_hash() is not None
        native_payload = native_model_payload(rs.request, repair_instruction=rs.repair_text)
        strands_payload = strands_model_payload(rs.request, repair_instruction=rs.repair_text)
        assert native_payload == strands_payload
        assert native_payload["repair_instruction"] == rs.repair_text
        assert "repair_prompt_version" in native_payload


def test_rp15_no_repair_context_for_non_repair_scenarios():
    for builder in (build_s0_ordinary_decision, build_s1_memory_decision, build_s3_tool_action_decision):
        request = builder(MODEL_ID, PROVIDER).request
        assert request.repair_context is None
        assert request.repair_facts_hash() is None


# ---------------------------------------------------------------------------
# RP16 - tool schema semantics: same logical action names / fields / enums
# ---------------------------------------------------------------------------


def test_rp16_tool_schema_logical_capability_parity():
    scenario = build_s3_tool_action_decision(MODEL_ID, PROVIDER)
    adapter = StrandsAgentAdapter("white")
    capture: list = []
    tool = adapter._make_tool(capture)

    # Same four logical fields Native's convert_response_to_action reads
    # from a raw response dict (action/target_id/trap_type/reason).
    result = tool(action="place_trap", target_id="near_object", trap_type="alarm", reason="test")
    assert isinstance(result, str)
    assert capture == [{"action": "place_trap", "target_id": "near_object", "trap_type": "alarm", "reason": "test"}]

    # Native's own textual tool-like representation names the identical
    # action vocabulary Strands' tool exposes.
    native_text = format_legal_actions_tool_like(scenario.legal_actions)
    assert "PLACE_TRAP(target_id)" in native_text
    for entry in scenario.legal_actions:
        assert _normalize_action_name(entry.get("action")) in VALID_ACTION_NAMES

    # place_trap_repair_hints (RP06's shared function) is the actual source
    # of legal trap_type enum values for BOTH runtimes - never a second,
    # independently-drifting enumeration. MULTIPLE_TYPES_REMAINING must
    # offer more than one, and the tool accepts each unchanged.
    legal_trap_types, _ = place_trap_repair_hints(scenario.observation, scenario.legal_actions)
    assert len(legal_trap_types) > 1, "S3 scenario must offer >1 legal trap_type (MULTIPLE_TYPES_REMAINING)"
    for trap_type in legal_trap_types:
        capture.clear()
        tool(action="place_trap", target_id="ambassador_safe", trap_type=trap_type, reason="test")
        assert capture[0]["trap_type"] == trap_type


# ---------------------------------------------------------------------------
# RP17 - output normalization parity
# ---------------------------------------------------------------------------


def test_rp17_output_normalization_parity_for_equivalent_raw_outputs():
    native_raw = {"action": "INSPECT", "target_id": "desk_01", "reason": "checking the desk"}
    strands_raw = {"action": "inspect", "target_id": "desk_01", "trap_type": None, "reason": "checking the desk"}
    native_norm = normalize_raw_output(native_raw)
    strands_norm = normalize_raw_output(strands_raw)
    assert native_norm["action"] == strands_norm["action"] == "inspect"
    assert native_norm["target_id"] == strands_norm["target_id"] == "desk_01"
    assert native_norm["visible_rationale"] == strands_norm["visible_rationale"] == "checking the desk"


def test_rp17_normalize_raw_output_handles_missing_action():
    assert normalize_raw_output({})["action"] is None


# ---------------------------------------------------------------------------
# RP18 - provider metadata field-set parity (presence, not value, parity)
# ---------------------------------------------------------------------------


def test_rp18_provider_metadata_field_set_parity():
    scenario = build_s0_ordinary_decision(MODEL_ID, PROVIDER)
    native_response = build_agent_response(
        raw_model_output={"action": "wait", "reason": "r"}, observation=scenario.observation,
        legal_actions=scenario.legal_actions, tool_calls=0,
        runtime_metadata={"runtime": "native"},
        provider_metadata={"latency_ms": 120.0, "input_tokens": 50, "output_tokens": 12, "model_id": MODEL_ID, "provider": PROVIDER},
    )
    strands_response = build_agent_response(
        raw_model_output={"action": "wait", "reason": "r"}, observation=scenario.observation,
        legal_actions=scenario.legal_actions, tool_calls=1,
        runtime_metadata={"runtime": "strands", "cycle_count": 1},
        provider_metadata={"latency_ms": 340.0, "input_tokens": 55, "output_tokens": 14, "model_id": MODEL_ID, "provider": PROVIDER},
    )
    common_keys = {"latency_ms", "input_tokens", "output_tokens", "model_id", "provider"}
    assert common_keys.issubset(native_response.provider_metadata.keys())
    assert common_keys.issubset(strands_response.provider_metadata.keys())
    assert native_response.validation_status == strands_response.validation_status == "valid"


# ---------------------------------------------------------------------------
# Phase B (spec section 24): canonical request equivalence across all four
# scenario families + Engine authority (spec sections 3, 5) untouched
# ---------------------------------------------------------------------------


def test_canonical_request_equivalence_for_all_four_scenario_families():
    s0 = build_s0_ordinary_decision(MODEL_ID, PROVIDER)
    s1 = build_s1_memory_decision(MODEL_ID, PROVIDER)
    s2 = build_s2_repair_decisions(MODEL_ID, PROVIDER)
    s3 = build_s3_tool_action_decision(MODEL_ID, PROVIDER)

    assert s0.request.metadata["scenario_family"] == S0_ORDINARY_DECISION
    assert s1.request.metadata["scenario_family"] == S1_MEMORY_DECISION
    assert all(r.request.metadata["scenario_family"] == S2_REPAIR_DECISION for r in s2)
    assert s3.request.metadata["scenario_family"] == S3_TOOL_ACTION_DECISION

    # S1 carries real new memory content S0 does not (RP13's own field, not incidental).
    assert s1.request.memory_hash() != s0.request.memory_hash()
    # S0 has no repair context; every S2 request does.
    assert s0.request.repair_context is None
    assert all(r.request.repair_context is not None for r in s2)


def test_engine_authority_untouched_building_canonical_requests_and_payloads():
    """Spec section 5: no runtime may mutate GameState/Observation directly.
    Observation is a frozen dataclass - if any step here touched it, this
    would raise FrozenInstanceError, not merely fail an equality check."""
    scenario = build_s3_tool_action_decision(MODEL_ID, PROVIDER)
    observation_before = copy.deepcopy(scenario.observation)
    native_model_payload(scenario.request)
    strands_model_payload(scenario.request)
    native_system_prompt(scenario.request)
    assert scenario.observation == observation_before


def test_native_system_prompt_includes_personality_when_present():
    from src.evaluation.agent_contract import build_agent_request

    scenario = build_s0_ordinary_decision(MODEL_ID, PROVIDER)
    with_personality = build_agent_request(
        actor_id=scenario.request.actor_id, episode_id=scenario.request.episode_id,
        observation=scenario.observation, legal_actions=scenario.legal_actions,
        memory_payload=scenario.request.memory_payload, personality_profile="cautious",
        model_id=MODEL_ID, provider_name=PROVIDER,
    )
    prompt_without = native_system_prompt(scenario.request)
    prompt_with = native_system_prompt(with_personality)
    assert prompt_without != prompt_with
    assert "cautious" in prompt_with.lower()
