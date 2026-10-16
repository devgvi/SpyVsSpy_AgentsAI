"""M6E.1 spec sections 3-5, 13: the five scenarios and their FactSpecs.

Full duplicate-count/location/format treatment (spec sections 7-9) runs on
S1, S2, S6 - each has a real, natural, already-present fact drawn from a
top-level `AgentMemory` list container (`known_locks`/`recent_failures`),
matching `duplicate_fact_variants.py`'s generic list-of-dicts model.

S0 (mandatory negative control) and S5 (one of the "additional snapshots
with strong lossless shifts") have no such real duplicable fact of their
own (S0's whole point is an empty, irrelevant memory; S5's own memory
facts are `recent_decisions`/`recent_interactions` about the SAME already-
visible target, not a cross-section duplicate) - spec section 5 already
anticipates this ("If duplicate or format manipulations alter it
strongly, report generic structural salience"). Their role here is
served by M6E's own already-collected C0/C1 data (reused, not re-run -
spec section 26 of M6E and this project's established "do not re-spend
live budget for identical information" discipline), which already is
exactly a FULL_FORMAT vs COMPACT_FORMAT comparison at each snapshot's
natural (unmanipulated) duplicate configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.duplicate_fact_variants import FactSpec
from src.evaluation.m6e_scenarios import (
    S1_RECENT_FAILED_ACTION,
    S2_KNOWN_LOCKED_OBJECT,
    S6_LONG_HISTORY,
    build_all_scenarios,
    build_compact_observation_payload,
)

FULL_TREATMENT_FAMILIES = (S1_RECENT_FAILED_ACTION, S2_KNOWN_LOCKED_OBJECT, S6_LONG_HISTORY)

FACT_SPECS: dict[str, FactSpec] = {
    S1_RECENT_FAILED_ACTION: FactSpec(
        category="RECENT_FAILURES", identity="filing_cabinet_01",
        canonical_value={"game_time": 8.0, "action": "search", "target_id": "filing_cabinet_01", "reason": "already_searched", "detail": "found nothing of value"},
        mem_container="recent_failures", mem_key="target_id",
    ),
    S2_KNOWN_LOCKED_OBJECT: FactSpec(
        category="KNOWN_LOCKS", identity="ambassador_safe",
        canonical_value={"target_id": "ambassador_safe", "lock_type": "code", "requires": "unknown", "satisfied": False},
        mem_container="known_locks", mem_key="target_id",
    ),
    S6_LONG_HISTORY: FactSpec(
        category="KNOWN_LOCKS", identity="archives_cabinet",
        canonical_value={"target_id": "archives_cabinet", "lock_type": "key", "requires": "requires archives_key", "satisfied": False},
        mem_container="known_locks", mem_key="target_id",
    ),
}


@dataclass(frozen=True)
class M6E1Scenario:
    scenario_id: str
    family: str
    observation: Any
    legal_actions: list
    observation_payload: dict[str, Any]
    memory_payload: dict[str, Any]
    fact: FactSpec


def build_full_treatment_scenarios() -> dict[str, M6E1Scenario]:
    base = build_all_scenarios()
    out = {}
    for family in FULL_TREATMENT_FAMILIES:
        s = base[family]
        obs = build_compact_observation_payload(s)
        out[family] = M6E1Scenario(s.scenario_id, family, s.observation, s.legal_actions, obs, s.memory_payload, FACT_SPECS[family])
    return out
