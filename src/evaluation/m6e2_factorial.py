"""M6E.2: the 2x2x2 representation-interaction factorial (spec sections
5-14). Reuses M6E.1's `duplicate_fact_variants.py` machinery unmodified -
5 of the 8 cells per snapshot are literally the same named conditions
M6E.1 already defined and (for S1/S2) already partly live-tested; only 3
new cells were added there to complete the cube.

Factorial axes (spec section 5):
    A - COPY COUNT:       A0 = SINGLE_COPY (1),      A1 = DOUBLE_COPY (2)
    B - DUPLICATE LOCATION: B0 = NATURAL_LOCATION,    B1 = AUXILIARY_LOCATION
    C - FORMAT:            C0 = FULL_FORMAT,          C1 = COMPACT_FORMAT

"NATURAL_LOCATION" always includes `memory` (the fact's real home);
"AUXILIARY_LOCATION" always includes `auxiliary` instead of `observation`
at double-copy, or is auxiliary-only at single-copy - `memory` is held
constant as the one location present in EVERY cell so B varies only the
SECOND section, never both the presence of `memory` and the "other"
section at once (the surgical M6E.1 discipline extended one level up).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.evaluation.duplicate_fact_variants import (
    A0B1C0_SINGLE_AUX_FULL,
    A0B1C1_SINGLE_AUX_COMPACT,
    A1B1C1_DOUBLE_AUX_COMPACT,
    D1_SINGLE_COPY,
    D2_DOUBLE_COPY,
    FactSpec,
    I1_SINGLE_COPY_COMPACT,
    L_MEM_AUX,
    R1_COMPACT_FORMAT,
    build_variant,
)
from src.evaluation.m6e1_scenarios import FACT_SPECS
from src.evaluation.m6e_scenarios import (
    S0_NO_HISTORY_RELEVANT,
    S1_RECENT_FAILED_ACTION,
    S2_KNOWN_LOCKED_OBJECT,
    build_all_scenarios,
    build_compact_observation_payload,
)

A0_SINGLE_COPY = "A0_SINGLE_COPY"
A1_DOUBLE_COPY = "A1_DOUBLE_COPY"
B0_NATURAL_LOCATION = "B0_NATURAL_LOCATION"
B1_AUXILIARY_LOCATION = "B1_AUXILIARY_LOCATION"
C0_FULL_FORMAT = "C0_FULL_FORMAT"
C1_COMPACT_FORMAT = "C1_COMPACT_FORMAT"

# (A, B, C) -> the M6E.1 condition name that realizes this cell.
FACTORIAL_CELLS: dict[tuple[str, str, str], str] = {
    (A0_SINGLE_COPY, B0_NATURAL_LOCATION, C0_FULL_FORMAT): D1_SINGLE_COPY,
    (A0_SINGLE_COPY, B0_NATURAL_LOCATION, C1_COMPACT_FORMAT): I1_SINGLE_COPY_COMPACT,
    (A0_SINGLE_COPY, B1_AUXILIARY_LOCATION, C0_FULL_FORMAT): A0B1C0_SINGLE_AUX_FULL,
    (A0_SINGLE_COPY, B1_AUXILIARY_LOCATION, C1_COMPACT_FORMAT): A0B1C1_SINGLE_AUX_COMPACT,
    (A1_DOUBLE_COPY, B0_NATURAL_LOCATION, C0_FULL_FORMAT): D2_DOUBLE_COPY,
    (A1_DOUBLE_COPY, B0_NATURAL_LOCATION, C1_COMPACT_FORMAT): R1_COMPACT_FORMAT,
    (A1_DOUBLE_COPY, B1_AUXILIARY_LOCATION, C0_FULL_FORMAT): L_MEM_AUX,
    (A1_DOUBLE_COPY, B1_AUXILIARY_LOCATION, C1_COMPACT_FORMAT): A1B1C1_DOUBLE_AUX_COMPACT,
}

# Condition name -> (A, B, C), the inverse lookup used by analysis.
CONDITION_TO_CELL: dict[str, tuple[str, str, str]] = {v: k for k, v in FACTORIAL_CELLS.items()}

PRIMARY_FAMILIES = (S2_KNOWN_LOCKED_OBJECT, S1_RECENT_FAILED_ACTION, S0_NO_HISTORY_RELEVANT)

# S0 has no memory-relevant fact of its own (spec section 31's whole point) -
# reuses S2's exact KNOWN_LOCKS:ambassador_safe FactSpec, injected via the
# same production `AgentMemory.record_known_lock`-shaped API M6C/M6E's own
# synthetic-memory snapshots already use, to test whether the SAME
# representation manipulations create salience even where the fact is not
# otherwise relevant to S0's own frozen decision.
S0_FACT = FACT_SPECS[S2_KNOWN_LOCKED_OBJECT]

FACT_SPECS_M6E2: dict[str, FactSpec] = {
    S2_KNOWN_LOCKED_OBJECT: FACT_SPECS[S2_KNOWN_LOCKED_OBJECT],
    S1_RECENT_FAILED_ACTION: FACT_SPECS[S1_RECENT_FAILED_ACTION],
    S0_NO_HISTORY_RELEVANT: S0_FACT,
}


@dataclass(frozen=True)
class M6E2Scenario:
    scenario_id: str
    family: str
    observation: Any
    legal_actions: list
    observation_payload: dict[str, Any]
    memory_payload: dict[str, Any]
    fact: FactSpec


def build_primary_scenarios() -> dict[str, M6E2Scenario]:
    base = build_all_scenarios()
    out = {}
    for family in PRIMARY_FAMILIES:
        s = base[family]
        obs = build_compact_observation_payload(s)
        out[family] = M6E2Scenario(s.scenario_id, family, s.observation, s.legal_actions, obs, s.memory_payload, FACT_SPECS_M6E2[family])
    return out


def build_cell(cell: tuple[str, str, str], scenario: M6E2Scenario):
    condition = FACTORIAL_CELLS[cell]
    return build_variant(condition, scenario.observation_payload, scenario.memory_payload, scenario.fact)


def duplicate_location_hash(locations: tuple[str, ...]) -> str:
    """Spec section 11-12: tracks placement, differs across location
    conditions holding count fixed - independent of count/format."""
    return hashlib.sha256(json.dumps(sorted(locations)).encode("utf-8")).hexdigest()


def format_hash(format_variant: str) -> str:
    """Spec section 11-12: tracks serialization choice alone."""
    return hashlib.sha256(format_variant.encode("utf-8")).hexdigest()
