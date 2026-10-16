"""M6B.1 spec sections 12-21: the Runtime Parity Invariant.

> Any behavior intended to be runtime-independent must have an executable
> parity test across supported runtime adapters.

This module holds the manifest data only (`RUNTIME_PARITY_MANIFEST`) -
the actual parity assertions live as real tests in
`tests/test_m6b1_runtime_parity.py`, referenced here by test id so the
manifest and the enforcement never drift apart silently. Parity does NOT
require identical internal loop/tool wrapper/latency/token count/provider
payload bytes (spec section 14) - those are legitimate runtime
differences, not audited here.

Sibling-fix audit (spec section 20): grepped the codebase for
equivalence-flavored comments ("same as Native", "identical contract",
"mirrors Native") - findings recorded per-entry below, each either
backed by a real test or corrected/qualified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PASS = "PASS"
FAIL = "FAIL"
RUNTIME_SPECIFIC = "RUNTIME_SPECIFIC"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class ParityEntry:
    entry_id: str
    description: str
    native_path: str
    strands_path: str
    test_id: str
    parity_required: bool
    status: str
    notes: str = ""


RUNTIME_PARITY_MANIFEST: tuple[ParityEntry, ...] = (
    ParityEntry(
        entry_id="RP01_PERSONALITY",
        description="A requested personality_profile reaches the actual model-facing system prompt for both actors.",
        native_path="Game._build_llm_agent -> LLMControllerConfig(personality_profile=...) -> LLMController._effective_system_prompt()",
        strands_path="Game._build_llm_agent -> StrandsAgentAdapter(personality_profile=...) -> self._effective_system_prompt (M6A fix)",
        test_id="tests/test_m6b_harness_invariants.py::test_personality_propagates_to_the_real_controller[*]",
        parity_required=True, status=PASS,
        notes="Fixed in M6A.2 (Game._build_llm_agent never passed personality_profile to Strands at all before that). Covered for both White and Black.",
    ),
    ParityEntry(
        entry_id="RP02_MODEL_CONFIG",
        description="Requested model ID and Bedrock region reach both adapters.",
        native_path="LLMController.model_name / self.model_client.region (BedrockClient)",
        strands_path="StrandsAgentAdapter.model_name / self._llm_region",
        test_id="tests/test_m6b_harness_invariants.py::test_model_id_propagates_to_both_actors[*], test_llm_region_propagates_to_both_actors[*]",
        parity_required=True, status=PASS,
        notes="Attribute location differs (a documented structural asymmetry, not a bug) - both accessors verified equal to the requested value.",
    ),
    ParityEntry(
        entry_id="RP03_ACTION_CONTRACT",
        description="Both runtimes recognize the same action-name set and normalize aliases identically.",
        native_path="VALID_ACTION_NAMES / _normalize_action_name (src/llm_controller.py)",
        strands_path="choose_action's own `action` argument, validated by the same validate_model_output()",
        test_id="tests/test_m6b1_runtime_parity.py::test_rp03_action_contract_parity",
        parity_required=True, status=PASS,
        notes="Native's own enum and Strands' tool argument both funnel into the one shared _normalize_action_name/VALID_ACTION_NAMES.",
    ),
    ParityEntry(
        entry_id="RP04_VALIDATION",
        description="An identical malformed/valid proposal is judged identically by both runtimes.",
        native_path="validate_model_output() (src/llm_controller.py)",
        strands_path="validate_model_output() - the exact same function, no Strands-side reimplementation",
        test_id="tests/test_m6b_place_trap_reliability.py::test_phase_c_* (6 cases), tests/test_m6b1_runtime_parity.py::test_rp04_validation_parity_matrix",
        parity_required=True, status=PASS,
        notes="Genuinely one shared function, not two independently-drifting implementations - the strongest parity guarantee in the whole comparison.",
    ),
    ParityEntry(
        entry_id="RP05_REPAIR_SEMANTICS",
        description="Both runtimes attempt at most one repair, using the same shared repair-instruction builder.",
        native_path="LLMController.config.max_retries=1 -> _repair_instruction()",
        strands_path="StrandsAgentAdapter.max_retries=1 -> _repair_instruction() (same function)",
        test_id="tests/test_m6b1_runtime_parity.py::test_rp05_repair_budget_parity",
        parity_required=True, status=PASS,
    ),
    ParityEntry(
        entry_id="RP06_LEGAL_TRAP_HINTS",
        description="A place_trap repair's own instruction text includes the actor's actual held trap types.",
        native_path="_request_action computes legal_trap_types from observation.trap_inventory, passes to _repair_instruction (M5F.3B.1)",
        strands_path="_request_action computes legal_trap_types identically, passes to _repair_instruction (M6B fix)",
        test_id="tests/test_m6b_place_trap_reliability.py::test_strands_repair_prompt_includes_legal_trap_values, test_strands_repair_prompt_fires_even_when_initial_error_is_missing_target",
        parity_required=True, status=PASS,
        notes="The M6B root-cause bug: Strands never passed legal_trap_types at all before this fix, despite an M5D.5 comment claiming 'identical repair contract to LLMController' - now true and test-backed.",
    ),
    ParityEntry(
        entry_id="RP07_FALLBACK",
        description="An unrecoverable decision falls back to the same {'action': 'wait'} shape on both runtimes.",
        native_path="executed_action_dict = {'action': 'wait'} when final_model_action_valid is False",
        strands_path="executed_action_dict = {'action': 'wait'} when final_model_action_valid is False (identical literal)",
        test_id="tests/test_m6b1_runtime_parity.py::test_rp07_fallback_shape_parity",
        parity_required=True, status=PASS,
    ),
    ParityEntry(
        entry_id="RP08_FAILURE_COUNTERS",
        description="provider_failure_count/validation_failure_count/repair_failure_count/fallback_count all exist and are kept separate on both runtimes.",
        native_path="LLMController (M5F.3B.1 split)",
        strands_path="StrandsAgentAdapter (M6A.2 split - did not exist at all before that milestone)",
        test_id="tests/test_m5f3b1_invalid_action_robustness.py::test_invalid_trap_type_increments_validation_not_provider_failure_count, tests/test_strands_agent.py::test_invalid_trap_type_increments_validation_not_provider_failure_count",
        parity_required=True, status=PASS,
        notes="Fixed in M6A.2 - Strands had only the combined fallback_count before that.",
    ),
    ParityEntry(
        entry_id="RP09_COMMUNICATION_FLAG",
        description="SOCIAL_ON/OFF communication policy injection.",
        native_path="LLMControllerConfig.communication_enabled -> schema/prompt communication block",
        strands_path="no communication-policy injection point exists at all",
        test_id="n/a - documented gap, not fixed (SOCIAL_OFF used throughout M6A/M6A.1/M6A.2/M6B/M6B.1)",
        parity_required=False, status=RUNTIME_SPECIFIC,
        notes="Already documented in M6A's own runtime_surface_audit.md. Out of scope to fix here (SOCIAL_ON is explicitly out of scope for M6B.1). Marked RUNTIME_SPECIFIC, not silently assumed equivalent.",
    ),
    ParityEntry(
        entry_id="RP10_MEMORY_FLAGS",
        description="Within-episode memory tracking, and the cross-episode-memory injection hook.",
        native_path="AgentMemoryTracker (shared class) for within-episode; LLMControllerConfig.model_payload_transform for cross-episode injection, applied once in LLMController._request_action",
        strands_path="AgentMemoryTracker (same shared class) for within-episode; StrandsAgentAdapter(model_payload_transform=...) (M6B.2 fix) applied once in StrandsAgentAdapter._request_action, same input/output/opportunity semantics",
        test_id="tests/test_m6b1_runtime_parity.py::test_rp10_within_episode_memory_parity, tests/test_m6b2_target_legality_and_payload_parity.py::test_rp10_model_payload_transform_now_exists_on_strands, test_rp10_no_transform_configured_is_byte_identical_to_before, test_rp10_transform_applied_once_covers_both_initial_and_repair, test_rp10_transform_hash_parity_same_logical_payload_same_hash",
        parity_required=True, status=PASS,
        notes="M6B.1 found this FAIL (model_payload_transform existed only on Native). M6B.2 spec sections 15-21 close it: StrandsAgentAdapter now accepts the same constructor parameter, Game._build_llm_agent now threads it into the Strands branch (see game.py), the hook is applied exactly once per decision to the same compact model-facing payload shape Native uses (covering both the initial and repair call, since Strands reuses the same `model_payload` object for both), default-off behavior is unchanged for every existing caller, and it cannot see hidden GameState (it only ever receives the already actor-visible compact payload). Not byte-identical provider payloads (spec section 17 explicitly does not require that) - verified at the shared logical-payload/hash level instead.",
    ),
    ParityEntry(
        entry_id="RP11_TELEMETRY",
        description="Both runtimes emit the llm_decision telemetry event through the same schema/field set.",
        native_path="LLMController._emit_decision_telemetry -> TelemetryLogger.emit_llm_decision",
        strands_path="StrandsAgentAdapter._emit_telemetry -> the same TelemetryLogger.emit_llm_decision",
        test_id="tests/test_m6b1_runtime_parity.py::test_rp11_telemetry_field_parity",
        parity_required=True, status=PASS,
        notes="Verified extensively across M6A/M6A.1/M6A.2's own analysis code, which reads observation_hash/legal_actions/executed_action/raw_response/repair_raw_response identically from both runtimes' events. M6B.2 adds same_invalid_action_signature_count as a new shared field (spec section 11), emitted identically by both.",
    ),
    ParityEntry(
        entry_id="RP12_LEGAL_TARGET_HINTS",
        description="For runtime-independent actions, both runtimes expose equivalent legal target information during repair.",
        native_path="LLMController._attempt_repair computes legal_place_trap_targets from the same legal_actions list, passes to _repair_instruction (M6B.2)",
        strands_path="StrandsAgentAdapter._request_action computes legal_place_trap_targets identically, passes to _repair_instruction (M6B.2, same shared function)",
        test_id="tests/test_m6b2_target_legality_and_payload_parity.py::test_rp12_repair_prompt_includes_legal_place_trap_targets_native, test_rp12_repair_prompt_includes_legal_place_trap_targets_strands, test_rp12_target_hint_matches_legal_actions_membership",
        parity_required=True, status=PASS,
        notes="New in M6B.2 (spec section 14), extending RP05/RP06's pattern to the target dimension - the exact gap the M6B.1 outlier episode (seed 9605) exposed: neither runtime's repair text previously named which targets (if any) were currently legal for PLACE_TRAP, only which trap types. Both runtimes now derive this from the identical, already-computed legal_actions list - never a second, independently-drifting computation.",
    ),
    # M6F spec section 16: RP13-RP18, appended below RP01-RP12 without
    # renumbering them (spec: "Do not renumber RP01-RP12"). These verify
    # the new canonical AgentRequest/AgentResponse boundary
    # (src/evaluation/agent_contract.py) itself, one layer above the
    # RP01-RP12 behavioral invariants above.
    ParityEntry(
        entry_id="RP13_AGENT_REQUEST_FIELDS",
        description="Native and Strands consume equivalent logical AgentRequest fields (observation, memory, personality, mission context, legal actions, repair context) from one canonical construction.",
        native_path="agent_contract.build_agent_request() -> agent_contract.native_model_payload()",
        strands_path="agent_contract.build_agent_request() -> agent_contract.strands_model_payload() (identical function - Strands' real _invoke_strands builds the same dict(model_payload)+repair_instruction shape)",
        test_id="tests/test_m6f_runtime_interoperability.py::test_rp13_native_and_strands_payloads_carry_identical_agent_request_fields",
        parity_required=True, status=PASS,
        notes="Formalizes what M6D already established empirically (the near-empty-payload bug fix) as an executable, hashable contract rather than an implicit per-driver convention.",
    ),
    ParityEntry(
        entry_id="RP14_GENERATION_SETTINGS",
        description="Model id, provider, and temperature propagate identically into the canonical request for both runtimes; both real adapters pin temperature=0.0.",
        native_path="AgentRequest.generation_settings (model_id, provider, temperature=0.0, max_tokens=None, thinking=None)",
        strands_path="StrandsAgentAdapter._build_default_agent's BedrockModel(temperature=0.0) - same fixed value, independently verified against the canonical request's own generation_settings",
        test_id="tests/test_m6f_runtime_interoperability.py::test_rp14_generation_settings_hash_stable_and_temperature_fixed",
        parity_required=True, status=PASS,
        notes="max_tokens/thinking are not currently configurable on either runtime (both None) - a real, symmetric absence, not a gap.",
    ),
    ParityEntry(
        entry_id="RP15_REPAIR_CONTEXT",
        description="Identical RepairFacts reach both runtimes before runtime serialization.",
        native_path="repair_prompt_variants.build_repair_facts() -> AgentRequest.repair_context -> agent_contract.native_model_payload()",
        strands_path="the same build_repair_facts()/render_contract() output -> AgentRequest.repair_context -> agent_contract.strands_model_payload() (identical function to native_model_payload)",
        test_id="tests/test_m6f_runtime_interoperability.py::test_rp15_repair_context_hash_identical_across_runtimes",
        parity_required=True, status=PASS,
        notes="Reuses RepairFacts/build_repair_facts from repair_prompt_variants.py unmodified (M6D.2 already established this as the canonical repair-fact source).",
    ),
    ParityEntry(
        entry_id="RP16_TOOL_SCHEMA_SEMANTICS",
        description="Native's format_legal_actions_tool_like text and Strands' choose_action tool expose the same logical action names, required fields, and enum values, even though the provider-facing structure (JSON block vs. real tool schema) differs.",
        native_path="format_legal_actions_tool_like(legal_actions) (src/llm_controller.py) - textual, ordinary JSON output",
        strands_path="StrandsAgentAdapter._make_tool's choose_action(action, target_id, trap_type, reason) - a real Strands tool schema",
        test_id="tests/test_m6f_runtime_interoperability.py::test_rp16_tool_schema_logical_capability_parity",
        parity_required=True, status=PASS,
        notes="Both funnel into the identical VALID_ACTION_NAMES/validate_model_output gate (RP03/RP04) - logical capability is provably equal even though representation is not, exactly as spec section 20 requires.",
    ),
    ParityEntry(
        entry_id="RP17_OUTPUT_NORMALIZATION",
        description="Equivalent raw model outputs from either runtime normalize to the same action/target_id/parameters/visible_rationale shape before validation.",
        native_path="agent_contract.normalize_raw_output() applied to LLMController's parsed model response",
        strands_path="agent_contract.normalize_raw_output() applied to the captured choose_action(...) proposal dict",
        test_id="tests/test_m6f_runtime_interoperability.py::test_rp17_output_normalization_parity_for_equivalent_raw_outputs",
        parity_required=True, status=PASS,
        notes="One shared normalization function, not two independently-drifting parsers (same HP02 discipline as RP04's shared validate_model_output).",
    ),
    ParityEntry(
        entry_id="RP18_PROVIDER_METADATA",
        description="Comparable telemetry fields (latency, input/output tokens, model id, provider, runtime) are recorded in a normalized shape for both runtimes, where available.",
        native_path="LLMController's provider-call telemetry (latency_ms, input_tokens, output_tokens) -> agent_contract.build_agent_response()'s provider_metadata",
        strands_path="_StrandsInvocation.latency_ms/input_tokens/output_tokens -> the same build_agent_response()'s provider_metadata",
        test_id="tests/test_m6f_runtime_interoperability.py::test_rp18_provider_metadata_field_set_parity",
        parity_required=True, status=PASS,
        notes="Field presence, not value equality, is what parity means here (spec section 22: 'normalize comparable telemetry fields... where available') - Strands' tool_call_count/cycle_count have no Native equivalent and are RUNTIME_SPECIFIC extras, not a gap.",
    ),
)


def manifest_as_dicts() -> list[dict]:
    return [
        {
            "id": e.entry_id, "description": e.description, "native_path": e.native_path, "strands_path": e.strands_path,
            "test_id": e.test_id, "parity_required": e.parity_required, "status": e.status, "notes": e.notes,
        }
        for e in RUNTIME_PARITY_MANIFEST
    ]


def parity_summary() -> dict:
    from collections import Counter
    counts = Counter(e.status for e in RUNTIME_PARITY_MANIFEST)
    return {
        "total_entries": len(RUNTIME_PARITY_MANIFEST),
        "counts_by_status": dict(counts),
        "any_required_fail": any(e.status == FAIL and e.parity_required for e in RUNTIME_PARITY_MANIFEST),
    }
