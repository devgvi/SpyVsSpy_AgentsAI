"""The frozen call-free policies (B1, B2-v1, B2-v2) return their golden outputs in the public tree.

The research archive computes the golden file on rows of the full Phase-A dataset (tests/test_final2_runtime.py,
skipped here). The public tree carries only those rows, in tests/data/final2_golden_rows.json, and recomputes the
same golden outputs with the same computation: the direct frozen loaders and the runtime policy registry must both
return the stored outputs, probability for probability.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spy_vs_ai.runtime import policy_registry as pr
from tests import test_final2_runtime as final2

ROOT = Path(__file__).resolve().parents[1]
ROWS = json.loads((ROOT / "tests/data/final2_golden_rows.json").read_text(encoding="utf-8"))


@pytest.fixture
def golden_rows(monkeypatch):
    from workshop.policy_emulator import dataset as ds
    assert ROWS["dataset"] == "phase-a-v2"
    monkeypatch.setattr(ds, "load", lambda dataset, *a, **k: list(ROWS["rows"]) if dataset == "phase-a-v2" else [])
    return ROWS["rows"]


def test_the_fixture_holds_the_golden_rows(golden_rows):
    golden = json.loads(final2.GOLDEN.read_text(encoding="utf-8"))
    assert [(r["decision_id"], r["key"]) for r in golden_rows] == [(r["decision_id"], r["key"]) for r in golden["rows"]]


def test_frozen_call_free_policies_return_the_golden_outputs(golden_rows):
    from workshop.policy_emulator import dataset as ds
    from workshop.policy_emulator import stage5b_protocol as s5b
    from workshop.policy_emulator.artifacts import load_policy
    expected = ds.load_provenance("phase-a-v2")["dataset_hash"]

    def direct(policy, key):
        if policy == "b1":
            return load_policy("phase-a-v2", key, emulator="table-v1", target="target-v2.1", expected_dataset_hash=expected)
        b2 = load_policy("phase-a-v2", key, emulator="logistic-v1", target="target-v2.1", expected_dataset_hash=expected)
        return s5b.arm_base("b2v2", b2) if policy == "b2_v2" else b2

    golden = json.loads(final2.GOLDEN.read_text(encoding="utf-8"))["outputs"]
    assert final2.frozen_outputs(direct) == golden
    assert final2.frozen_outputs(lambda policy, key: pr.load_frozen(policy, key).base) == golden
