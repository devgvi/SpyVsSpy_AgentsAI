"""M6F spec sections 25-27, 37-39: the four controlled scenario families,
built entirely from reused, already-proven snapshots (spec section 26 -
"Do not create a totally new world unless necessary"). No new GameState is
invented for this milestone.

S0 ORDINARY_DECISION      -> memory_snapshot_set's MULTIPLE_LEGAL_TARGETS, empty memory (spec section 39's "no repair, no special memory manipulation" control)
S1 MEMORY_DECISION        -> memory_snapshot_set's RECENT_FAILED_ACTION (spec section 38's memory-positive control)
S2 REPAIR_DECISION        -> recovery_corpus's INVALID_TRAP_TYPE + MISSING_REQUIRED_FIELD, CURRENT_PRODUCTION + NEUTRAL_STRUCTURED contracts only (spec sections 37, 48)
S3 TOOL_ACTION_DECISION   -> trap_snapshot_set's MULTIPLE_TYPES_REMAINING (a real place_trap tool-argument choice - trap_type enum selection)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src import constants
from src.evaluation.agent_contract import AgentRequest, build_agent_request
from src.evaluation.memory_snapshot_set import MULTIPLE_LEGAL_TARGETS, RECENT_FAILED_ACTION, build_memory_snapshot_set
from src.evaluation.recovery_corpus import INVALID_TRAP_TYPE, MISSING_REQUIRED_FIELD, build_invalid_action_corpus
from src.evaluation.repair_contracts import N0_CURRENT_PRODUCTION, N2_NEUTRAL_STRUCTURED, P0_CANONICAL, render_contract
from src.evaluation.repair_prompt_variants import build_repair_facts
from src.evaluation.trap_snapshot_set import MULTIPLE_TYPES_REMAINING, build_trap_snapshot_set
from src.llm.agent_memory import AgentMemory
from src.llm_controller import validate_model_output

S0_ORDINARY_DECISION = "S0_ORDINARY_DECISION"
S1_MEMORY_DECISION = "S1_MEMORY_DECISION"
S2_REPAIR_DECISION = "S2_REPAIR_DECISION"
S3_TOOL_ACTION_DECISION = "S3_TOOL_ACTION_DECISION"
SCENARIO_FAMILIES = (S0_ORDINARY_DECISION, S1_MEMORY_DECISION, S2_REPAIR_DECISION, S3_TOOL_ACTION_DECISION)

SELECTED_REPAIR_FAILURE_FAMILIES = (INVALID_TRAP_TYPE, MISSING_REQUIRED_FIELD)
SELECTED_REPAIR_CONTRACTS = (N0_CURRENT_PRODUCTION, N2_NEUTRAL_STRUCTURED)

ACTOR_ID = constants.WHITE_SPY_ID


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    observation: Any
    legal_actions: list
    request: AgentRequest


@dataclass(frozen=True)
class RepairScenario:
    scenario_id: str
    family: str
    case_id: str
    failure_family: str
    contract: str
    observation: Any
    legal_actions: list
    invalid_raw_response: dict
    repair_text: str
    request: AgentRequest


def _empty_memory_payload() -> dict[str, Any]:
    return AgentMemory(ACTOR_ID).to_compact_dict()


def build_s0_ordinary_decision(model_id: str, provider_name: str) -> Scenario:
    snap = next(s for s in build_memory_snapshot_set() if s.category == MULTIPLE_LEGAL_TARGETS)
    request = build_agent_request(
        actor_id=ACTOR_ID, episode_id="m6f-s0-ordinary-decision", observation=snap.observation,
        legal_actions=snap.legal_actions, memory_payload=_empty_memory_payload(), personality_profile=None,
        model_id=model_id, provider_name=provider_name, repair_context=None,
        metadata={"scenario_family": S0_ORDINARY_DECISION, "source_snapshot": snap.snapshot_id},
    )
    return Scenario(snap.snapshot_id, S0_ORDINARY_DECISION, snap.observation, snap.legal_actions, request)


def build_s1_memory_decision(model_id: str, provider_name: str) -> Scenario:
    snap = next(s for s in build_memory_snapshot_set() if s.category == RECENT_FAILED_ACTION)
    request = build_agent_request(
        actor_id=ACTOR_ID, episode_id="m6f-s1-memory-decision", observation=snap.observation,
        legal_actions=snap.legal_actions, memory_payload=snap.synthetic_memory, personality_profile=None,
        model_id=model_id, provider_name=provider_name, repair_context=None,
        metadata={"scenario_family": S1_MEMORY_DECISION, "source_snapshot": snap.snapshot_id},
    )
    return Scenario(snap.snapshot_id, S1_MEMORY_DECISION, snap.observation, snap.legal_actions, request)


def build_s3_tool_action_decision(model_id: str, provider_name: str) -> Scenario:
    snap = next(s for s in build_trap_snapshot_set() if s.category == MULTIPLE_TYPES_REMAINING)
    request = build_agent_request(
        actor_id=ACTOR_ID, episode_id="m6f-s3-tool-action-decision", observation=snap.observation,
        legal_actions=snap.legal_actions, memory_payload=_empty_memory_payload(), personality_profile=None,
        model_id=model_id, provider_name=provider_name, repair_context=None,
        metadata={"scenario_family": S3_TOOL_ACTION_DECISION, "source_snapshot": snap.snapshot_id},
    )
    return Scenario(snap.snapshot_id, S3_TOOL_ACTION_DECISION, snap.observation, snap.legal_actions, request)


def build_s2_repair_decisions(model_id: str, provider_name: str) -> list[RepairScenario]:
    corpus = {c.failure_family: c for c in build_invalid_action_corpus() if c.failure_family in SELECTED_REPAIR_FAILURE_FAMILIES}
    out: list[RepairScenario] = []
    for family, case in corpus.items():
        v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
        facts = build_repair_facts(case, v)
        for contract in SELECTED_REPAIR_CONTRACTS:
            repair_text = render_contract(contract, P0_CANONICAL, facts, case, v)
            repair_context = {"facts": asdict(facts), "contract": contract, "perturbation": P0_CANONICAL}
            request = build_agent_request(
                actor_id=ACTOR_ID, episode_id=f"m6f-s2-repair-{case.case_id}-{contract}", observation=case.observation,
                legal_actions=case.legal_actions, memory_payload=_empty_memory_payload(), personality_profile=None,
                model_id=model_id, provider_name=provider_name, repair_context=repair_context,
                metadata={"scenario_family": S2_REPAIR_DECISION, "source_case": case.case_id, "contract": contract},
            )
            out.append(RepairScenario(
                f"{case.case_id}__{contract}", S2_REPAIR_DECISION, case.case_id, family, contract,
                case.observation, case.legal_actions, case.invalid_raw_response, repair_text, request,
            ))
    return out


def build_all_scenarios(model_id: str, provider_name: str) -> dict[str, Any]:
    return {
        S0_ORDINARY_DECISION: build_s0_ordinary_decision(model_id, provider_name),
        S1_MEMORY_DECISION: build_s1_memory_decision(model_id, provider_name),
        S2_REPAIR_DECISION: build_s2_repair_decisions(model_id, provider_name),
        S3_TOOL_ACTION_DECISION: build_s3_tool_action_decision(model_id, provider_name),
    }
