"""M6E.1: surgical duplicate-count / duplicate-location / compact-format
variant construction for ONE real, already-present fact per snapshot
(spec sections 6-14).

Three real sections are used identically for every manipulated fact (spec
section 8's own example list - "Observation, AgentMemory, auxiliary
context"):

    "memory"      - the fact's natural AgentMemory home (already exists
                    in production for every fact type used here)
    "observation" - `observation_payload["memory"][<key>]`, the REAL,
                    already-precedented "Observation's echo of an
                    AgentMemory fact" sub-block (this exact section
                    already carries `visited_rooms`/`known_locked_objects`
                    in production - extended here to also carry the
                    other fact types under test, not invented from
                    nothing)
    "auxiliary"   - `observation_payload["auxiliary_context"][<key>]`,
                    a new top-level section - explicitly sanctioned by
                    spec section 8's own example list, used identically
                    for every fact (never a fact-specific invented slot)

Only ONE fact is ever manipulated per experiment; every other section of
the payload is passed through unchanged from the real production shapes
built by `m6e_scenarios.build_compact_observation_payload()` +
`AgentMemory.to_compact_dict()`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

MEMORY = "memory"
OBSERVATION = "observation"
AUXILIARY = "auxiliary"

# Real, already-existing production keys inside `observation_payload["memory"]`
# (compact_observation_for_model's own echo block) - e.g. KNOWN_LOCKS's real
# echo lives at `known_locked_objects` (id-only), not a key spelled after
# the category name. A fact with no such existing key (RECENT_FAILURES has
# no observation-side echo in production at all) falls back to a new,
# honestly-synthetic slot named after its category - documented, not
# silently treated as equivalent to a real field.
_OBSERVATION_ECHO_KEYS: dict[str, str] = {
    "KNOWN_LOCKS": "known_locked_objects",
    "VISITED_ROOMS": "visited_rooms",
}

D1_SINGLE_COPY = "D1_SINGLE_COPY"
D2_DOUBLE_COPY = "D2_DOUBLE_COPY"
D3_TRIPLE_COPY = "D3_TRIPLE_COPY"
L_OBS_AUX = "L_OBS_AUX"
L_MEM_AUX = "L_MEM_AUX"
R1_COMPACT_FORMAT = "R1_COMPACT_FORMAT"
I1_SINGLE_COPY_COMPACT = "I1_SINGLE_COPY_COMPACT"
# M6E.2's own three missing cells to complete the 2x2x2 factorial cube
# (spec sections 5-6) - the other 5 cells already exist above.
A0B1C0_SINGLE_AUX_FULL = "A0B1C0_SINGLE_AUX_FULL"
A0B1C1_SINGLE_AUX_COMPACT = "A0B1C1_SINGLE_AUX_COMPACT"
A1B1C1_DOUBLE_AUX_COMPACT = "A1B1C1_DOUBLE_AUX_COMPACT"

FULL_FORMAT = "FULL_FORMAT"
COMPACT_FORMAT = "COMPACT_FORMAT"

# Each named condition: (locations, format). D2_DOUBLE_COPY is deliberately
# also today's natural production placement (memory+observation) AND the
# "R0_FULL_FORMAT" baseline for the format axis - one condition serving
# three roles rather than three separate live-call cells for the same
# underlying payload (spec section 11: one axis at a time, not a large
# factorial).
CONDITIONS: dict[str, tuple[tuple[str, ...], str]] = {
    D1_SINGLE_COPY: ((MEMORY,), FULL_FORMAT),
    D2_DOUBLE_COPY: ((MEMORY, OBSERVATION), FULL_FORMAT),
    D3_TRIPLE_COPY: ((MEMORY, OBSERVATION, AUXILIARY), FULL_FORMAT),
    L_OBS_AUX: ((OBSERVATION, AUXILIARY), FULL_FORMAT),
    L_MEM_AUX: ((MEMORY, AUXILIARY), FULL_FORMAT),
    R1_COMPACT_FORMAT: ((MEMORY, OBSERVATION), COMPACT_FORMAT),
    # spec section 28's interaction gate: only run if no single axis
    # explains S2 - completes a 2x2 (copy count 1/2 x format full/compact)
    # alongside D1_SINGLE_COPY and D2_DOUBLE_COPY/R1_COMPACT_FORMAT.
    I1_SINGLE_COPY_COMPACT: ((MEMORY,), COMPACT_FORMAT),
    A0B1C0_SINGLE_AUX_FULL: ((AUXILIARY,), FULL_FORMAT),
    A0B1C1_SINGLE_AUX_COMPACT: ((AUXILIARY,), COMPACT_FORMAT),
    A1B1C1_DOUBLE_AUX_COMPACT: ((MEMORY, AUXILIARY), COMPACT_FORMAT),
}

# spec section 9: R0 is the natural D2 placement under FULL_FORMAT.
R0_FULL_FORMAT = D2_DOUBLE_COPY

_C1_REDUNDANT_KEYS = ("available_actions_text",)


@dataclass(frozen=True)
class FactSpec:
    """One real, already-present fact (spec section 13) to manipulate.
    `mem_path`/`obs_echo_key` describe where this fact naturally lives in
    `memory_payload` / the real `observation_payload["memory"]` echo
    block - `remove_fn`/`insert_mem_fn` know how to cleanly take it out or
    put it back, since removal must be exact (never leave a stray empty
    container behind that would itself be a representation difference)."""

    category: str
    identity: str
    canonical_value: Any  # the single, byte-identical value placed at every location
    mem_container: str  # top-level key in memory_payload holding this fact's list, or "" for a dict field
    mem_key: str  # for list containers: the field within each ITEM used to match `identity` (e.g. "target_id"); for dict fields: "" (whole container is the fact)


def _get_mem_items(mem: dict[str, Any], fact: FactSpec) -> list:
    return mem.get(fact.mem_container, []) if fact.mem_container else []


def _remove_from_memory(mem: dict[str, Any], fact: FactSpec) -> None:
    if not fact.mem_container:
        mem.pop(fact.category.lower(), None)
        return
    items = mem.get(fact.mem_container, [])
    mem[fact.mem_container] = [it for it in items if it.get(fact.mem_key) != fact.identity]


def _insert_into_memory(mem: dict[str, Any], fact: FactSpec) -> None:
    _remove_from_memory(mem, fact)
    if not fact.mem_container:
        return
    mem.setdefault(fact.mem_container, []).append(copy.deepcopy(fact.canonical_value))


def _observation_echo_key(fact: FactSpec) -> str:
    return _OBSERVATION_ECHO_KEYS.get(fact.category, fact.category.lower())


def _remove_from_observation_echo(obs: dict[str, Any], fact: FactSpec) -> None:
    """Removes ONLY this fact's own id from the real echo list, preserving
    any other unrelated entries already there (e.g. S6's real
    `door_security_corridor` lock echo must survive untouched when this
    function targets `archives_cabinet`) - never a wholesale key
    delete/replace."""
    key = _observation_echo_key(fact)
    memory_block = obs.setdefault("memory", {})
    existing = memory_block.get(key, [])
    memory_block[key] = [x for x in existing if x != fact.identity]


def _insert_into_observation_echo(obs: dict[str, Any], fact: FactSpec) -> None:
    """Matches the REAL production shape for a known echo key (id-only,
    e.g. `known_locked_objects`) rather than inventing a richer parallel
    key - spec section 7's 'byte-identical where possible' explicitly
    allows for cases where natural section shapes differ, and this is
    exactly that case (id-only is Observation's real convention here)."""
    key = _observation_echo_key(fact)
    memory_block = obs.setdefault("memory", {})
    existing = [x for x in memory_block.get(key, []) if x != fact.identity]
    if fact.category in _OBSERVATION_ECHO_KEYS:
        memory_block[key] = existing + [fact.identity]
    else:
        # No real precedent for this category - a synthetic slot, kept
        # byte-identical to the canonical value like `auxiliary` (never a
        # partial id-only echo invented for a section that has no real
        # shape to match at all).
        memory_block[key] = existing + [copy.deepcopy(fact.canonical_value)]


def _remove_from_auxiliary(obs: dict[str, Any], fact: FactSpec) -> None:
    aux = obs.get("auxiliary_context")
    if aux and fact.category.lower() in aux:
        del aux[fact.category.lower()]
        if not aux:
            obs.pop("auxiliary_context", None)


def _insert_into_auxiliary(obs: dict[str, Any], fact: FactSpec) -> None:
    _remove_from_auxiliary(obs, fact)
    obs.setdefault("auxiliary_context", {})[fact.category.lower()] = [copy.deepcopy(fact.canonical_value)]


@dataclass
class VariantResult:
    condition: str
    observation_payload: dict[str, Any]
    memory_payload: dict[str, Any]
    copy_count: int
    locations: tuple[str, ...]
    format_variant: str

    def to_json(self) -> str:
        payload = {**self.observation_payload, "strategy_memory": self.memory_payload}
        if self.format_variant == COMPACT_FORMAT:
            payload = dict(payload)
            for k in _C1_REDUNDANT_KEYS:
                payload.pop(k, None)
            return json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return json.dumps(payload, sort_keys=True)


def build_variant(condition: str, observation_payload: dict[str, Any], memory_payload: dict[str, Any], fact: FactSpec) -> VariantResult:
    locations, format_variant = CONDITIONS[condition]
    obs = copy.deepcopy(observation_payload)
    mem = copy.deepcopy(memory_payload)

    # Start from a clean slate: the fact is absent everywhere, then placed
    # into exactly the requested locations - never a partial/leftover copy.
    _remove_from_memory(mem, fact)
    _remove_from_observation_echo(obs, fact)
    _remove_from_auxiliary(obs, fact)

    if MEMORY in locations:
        _insert_into_memory(mem, fact)
    if OBSERVATION in locations:
        _insert_into_observation_echo(obs, fact)
    if AUXILIARY in locations:
        _insert_into_auxiliary(obs, fact)

    return VariantResult(condition, obs, mem, len(locations), locations, format_variant)


def unique_fact_hash(observation_payload: dict[str, Any], memory_payload: dict[str, Any], fact: FactSpec) -> str:
    """Ignores multiplicity/location - true whenever the fact's canonical
    value is present in at least one of the three sections (spec section
    17-18)."""
    echo_key = _observation_echo_key(fact)
    observation_echo = observation_payload.get("memory", {}).get(echo_key, [])
    if fact.category in _OBSERVATION_ECHO_KEYS:
        observation_present = fact.identity in observation_echo
    else:
        observation_present = any(item == fact.canonical_value for item in observation_echo)
    present = bool(
        any(it.get(fact.mem_key) == fact.identity for it in _get_mem_items(memory_payload, fact))
        or observation_present
        or fact.category.lower() in observation_payload.get("auxiliary_context", {})
    )
    canonical = {"category": fact.category, "identity": fact.identity, "present": present}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def multiset_fact_hash(copy_count: int, locations: tuple[str, ...], fact: FactSpec) -> str:
    """Includes duplicate count AND which sections hold it (spec section
    17-18) - differs whenever count or location set differs."""
    canonical = {"category": fact.category, "identity": fact.identity, "copy_count": copy_count, "locations": sorted(locations)}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def representation_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def background_fact_payload(observation_payload: dict[str, Any], memory_payload: dict[str, Any], fact: FactSpec) -> tuple[dict, dict]:
    """The rest of the payload with the manipulated fact's own natural
    homes cleared (so `request_fact_model.extract_request_facts` on this
    output, compared across every variant of the SAME snapshot, verifies
    'every OTHER unique fact stayed identical' - spec section 6)."""
    obs = copy.deepcopy(observation_payload)
    mem = copy.deepcopy(memory_payload)
    _remove_from_memory(mem, fact)
    _remove_from_observation_echo(obs, fact)
    _remove_from_auxiliary(obs, fact)
    return obs, mem
