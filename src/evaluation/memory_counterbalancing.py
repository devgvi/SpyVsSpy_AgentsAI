"""M6C.1: 2x2 factorial memory-representation counterbalancing (spec
sections 1-3).

M6C found a representation effect (M1 vs M2, same facts) but its negative
control revealed a confound: even ZERO facts, different shell structure,
shifted behavior. M6C.1 separates FACT CONTENT (F0 empty / F1 real) from
REPRESENTATION (R0 current-shape / R1 structured-shape) as two independent
axes:

    C0 EMPTY_CURRENT      = F0 x R0  (empty facts, M1's own shape)
    C1 EMPTY_STRUCTURED   = F0 x R1  (empty facts, M2's own shape)
    C2 FACTS_CURRENT      = F1 x R0  (real facts,  M1's own shape) - identical to M6C's M1
    C3 FACTS_STRUCTURED   = F1 x R1  (real facts,  M2's own shape) - identical to M6C's M2

Deliberately reuses `memory_representation.py`'s own
`frozen_condition_transform`/`_structured_facts_from_compact` unmodified
(spec section 5: "Do not add new memory fields, strategic hints,
summaries, or policy text") - C2/C3 are LITERALLY M6C's M1/M2 conditions,
just renamed for this factorial's own axis labels; C0/C1 are the exact
same two shapes (`frozen_condition_transform`'s M1_CURRENT_MEMORY/
M2_STRUCTURED_FACTS branches) applied to a genuinely fresh, empty
`AgentMemory` instead of a populated one - no new representation was
invented for this milestone.
"""

from __future__ import annotations

from typing import Any

from src import constants
from src.evaluation.memory_representation import (
    M1_CURRENT_MEMORY,
    M2_STRUCTURED_FACTS,
    extract_memory_facts,
    frozen_condition_transform,
    memory_fact_hash,
)
from src.llm.agent_memory import AgentMemory

C0_EMPTY_CURRENT = "C0_EMPTY_CURRENT"
C1_EMPTY_STRUCTURED = "C1_EMPTY_STRUCTURED"
C2_FACTS_CURRENT = "C2_FACTS_CURRENT"
C3_FACTS_STRUCTURED = "C3_FACTS_STRUCTURED"
COUNTERBALANCING_CONDITIONS = (C0_EMPTY_CURRENT, C1_EMPTY_STRUCTURED, C2_FACTS_CURRENT, C3_FACTS_STRUCTURED)

# Which fact source (empty vs. real) and which underlying M6C
# representation shape (M1 vs. M2) each condition maps to (spec section 2's
# A/B axes, kept explicit rather than inferred from the name string).
_CONDITION_AXES = {
    C0_EMPTY_CURRENT: ("F0", "R0", M1_CURRENT_MEMORY),
    C1_EMPTY_STRUCTURED: ("F0", "R1", M2_STRUCTURED_FACTS),
    C2_FACTS_CURRENT: ("F1", "R0", M1_CURRENT_MEMORY),
    C3_FACTS_STRUCTURED: ("F1", "R1", M2_STRUCTURED_FACTS),
}


def empty_agent_memory_dict(actor_id: str = constants.WHITE_SPY_ID) -> dict[str, Any]:
    """Spec section 10: a genuinely, semantically empty AgentMemory - no
    hidden statements like "no rooms visited" beyond what a truly fresh
    AgentMemory naturally produces (empty lists/dicts throughout, exactly
    matching AgentMemory.__init__'s own defaults - never a hand-built
    dict)."""
    return AgentMemory(actor_id).to_compact_dict()


# Spec section 12: both C0 and C1's empty fact set must equal this one
# canonical hash - computed once, actor-id-independent (AgentMemory's
# `actor_id` field is never itself a "fact" extract_memory_facts reads).
CANONICAL_EMPTY_FACT_HASH = memory_fact_hash(empty_agent_memory_dict("canonical"))


def counterbalancing_transform(condition: str, real_synthetic_m1_dict: dict[str, Any], actor_id: str = constants.WHITE_SPY_ID):
    """Returns the `model_payload_transform` for one of the 4 factorial
    conditions, reusing `frozen_condition_transform` unchanged (spec
    section 5)."""
    if condition not in _CONDITION_AXES:
        raise ValueError(f"unknown counterbalancing condition {condition!r}")
    fact_axis, _repr_axis, base_condition = _CONDITION_AXES[condition]
    source_dict = empty_agent_memory_dict(actor_id) if fact_axis == "F0" else real_synthetic_m1_dict
    return frozen_condition_transform(base_condition, source_dict)


def condition_axes(condition: str) -> tuple[str, str]:
    """(fact_axis, representation_axis) for a condition - "F0"/"F1",
    "R0"/"R1"."""
    fact_axis, repr_axis, _ = _CONDITION_AXES[condition]
    return fact_axis, repr_axis


def condition_payload(condition: str, real_synthetic_m1_dict: dict[str, Any], actor_id: str = constants.WHITE_SPY_ID) -> dict[str, Any]:
    """The actual `strategy_memory` dict for a condition, for direct
    hashing/auditing without running a trial."""
    transform = counterbalancing_transform(condition, real_synthetic_m1_dict, actor_id)
    return transform({"strategy_memory": {}})["strategy_memory"]
