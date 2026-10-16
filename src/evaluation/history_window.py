"""M6E.3: deterministic chronological history-window truncation, with
representation otherwise completely frozen (spec section 5 - current
canonical production representation, natural duplicate locations,
full-format serialization; no auxiliary placement, no compact format).

Window set (spec section 6): the real `S6_LONG_HISTORY` snapshot's own
history-bearing lists top out at 5 entries (`recent_decisions`) - fewer
than the spec's own illustrative 8/5/3/1/0 ladder - so per spec's own
accommodation ("choose the largest meaningful windows available and
document them"), windows are H_FULL/H4/H3/H2/H1/H0 rather than
H_FULL/H8/H5/H3/H1/H0: H4 and H3 each remove exactly one further boundary
(`recent_decisions` alone, then `recent_decisions`+`recent_interactions`),
and H3->H2 is exactly where `S6`'s own known failure event
(`filing_cabinet_01`, the oldest of 3 recorded failures) drops out of
`recent_failures` - documented in `history_manifest.json`, not assumed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

H_FULL = "H_FULL"
H4 = "H4"
H3 = "H3"
H2 = "H2"
H1 = "H1"
H0 = "H0"

# cap=None means "no truncation" (H_FULL). Ordered largest to smallest.
WINDOWS: dict[str, int | None] = {H_FULL: None, H4: 4, H3: 3, H2: 2, H1: 1, H0: 0}
WINDOW_ORDER = (H_FULL, H4, H3, H2, H1, H0)

HISTORY_FIELDS = ("recent_interactions", "recent_failures", "recent_events", "recent_decisions")


@dataclass
class WindowResult:
    window: str
    cap: int | None
    memory_payload: dict[str, Any]
    removed_events: dict[str, list[dict]]  # per field, the entries dropped relative to H_FULL


def apply_window(memory_payload: dict[str, Any], window: str) -> WindowResult:
    cap = WINDOWS[window]
    mem = copy.deepcopy(memory_payload)
    removed: dict[str, list[dict]] = {}
    for field in HISTORY_FIELDS:
        items = mem.get(field, [])
        if cap is None or len(items) <= cap:
            removed[field] = []
            continue
        kept = items[-cap:] if cap > 0 else []
        removed[field] = items[: len(items) - cap] if cap > 0 else list(items)
        mem[field] = kept
    return WindowResult(window, cap, mem, removed)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def history_full_hash(memory_payload: dict[str, Any]) -> str:
    """Hash of the untouched H_FULL history - a fixed reference point."""
    return _hash({f: memory_payload.get(f, []) for f in HISTORY_FIELDS})


def history_window_hash(window_memory_payload: dict[str, Any]) -> str:
    """Hash of exactly what a given window retains."""
    return _hash({f: window_memory_payload.get(f, []) for f in HISTORY_FIELDS})


def history_fact_hash(window_memory_payload: dict[str, Any]) -> str:
    """Order-INsensitive: the set of retained (field, entry) facts,
    regardless of position - differs from history_sequence_hash only when
    an add-back test reorders identical retained content (never expected
    here, since add-back always restores original chronological position,
    but kept distinct per spec section 14's own vocabulary)."""
    facts = []
    for f in HISTORY_FIELDS:
        for entry in window_memory_payload.get(f, []):
            facts.append((f, json.dumps(entry, sort_keys=True, default=str)))
    return _hash(sorted(facts))


def history_sequence_hash(window_memory_payload: dict[str, Any]) -> str:
    """Order-sensitive (spec section 14): differs if the same retained
    facts appeared in a different order."""
    return _hash({f: [json.dumps(e, sort_keys=True, default=str) for e in window_memory_payload.get(f, [])] for f in HISTORY_FIELDS})


def event_manifest(memory_payload: dict[str, Any]) -> list[dict]:
    """Spec section 13: one record per real history event, in original
    chronological position, with presence flags per window."""
    manifest = []
    for field in HISTORY_FIELDS:
        items = memory_payload.get(field, [])
        n = len(items)
        for i, entry in enumerate(items):
            age_from_newest = n - i  # 1 = most recent
            manifest.append({
                "event_index": i, "field": field, "event": entry,
                "timestamp": entry.get("game_time"), "event_type": field,
                "actor_visible": True,
                "present_in_window": {
                    w: (cap is None or age_from_newest <= cap) for w, cap in WINDOWS.items()
                },
            })
    return manifest
