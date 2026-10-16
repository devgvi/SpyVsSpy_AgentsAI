"""The public repository itself: its documents, its research statements, its frozen artifacts and its hygiene.
Call-free and offline."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from spy_vs_ai.runtime import policy_registry as pr

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = json.loads((ROOT / "PUBLIC_EXPORT_PROVENANCE.json").read_text(encoding="utf-8"))
DOCS = ("README.md", "CHANGELOG.md", "docs/README.md", "docs/play/PLAYING_THE_GAME.md",
        "docs/play/PROVIDERS_AND_MODELS.md", "docs/research/STORY.md", "docs/research/RESULTS.md",
        "docs/project/ARCHITECTURE.md", "docs/project/GLOSSARY.md", "paper/SPY_VS_AI_PAPER.md",
        "examples/replays/README.md", "assets/README.md")
LOCAL_OUTPUTS = ("runs", "replays", "runtime_logs")  # your own traces: ignored by Git, may exist locally


def read(rel: str) -> str:
    """A document with its line breaks folded, so phrases are found across wrapped lines."""
    return " ".join((ROOT / rel).read_text(encoding="utf-8").split())


@pytest.fixture(scope="module")
def validator():
    sys.path.insert(0, str(ROOT))
    from tools.validate_public_repo import Validator
    return Validator(ROOT, skip_tests=True, python=sys.executable)


def test_documents_exist_and_there_are_few_of_them():
    for rel in DOCS:
        assert (ROOT / rel).is_file() and len(read(rel)) > 200, rel
    assert len(list((ROOT / "docs").rglob("*.md"))) == 7  # the guide + 6 pages: no scattered fragments


def test_readme_covers_the_public_questions_and_sends_newcomers_to_the_guide():
    readme = read("README.md")
    for needle in ("Start with [the guide](docs/README.md)", "## Install", "## Quick start", "python -m spy_vs_ai.play",
                   "call-free", "Providers:", "Research mode:", "Replays:", "## What the project learned",
                   "paper", "## Reproducing the research", "docs/play/PLAYING_THE_GAME.md"):
        assert needle in readme, needle


def test_the_guide_presents_the_project_before_any_policy_id():
    guide = read("docs/README.md")
    for heading in ("## 1. The game in two minutes", "## 2. Who can play a spy", "## 3. The interface",
                    "### The setup screen", "### The match screen", "### Views and keys", "## 4. The research in brief",
                    "## 5. Where to go next"):
        assert heading in guide, heading
    for policy in pr.POLICY_IDS:
        assert f"`{policy}`" in guide, policy
    # the game and the players are explained before the first policy id appears
    assert guide.index("## 1. The game in two minutes") < guide.index("## 2. Who can play a spy") < guide.index("`b2_v1`")
    for image in ("media/setup_screen.png", "media/hero_door_trap.png", "media/overview.png", "media/research_view.png",
                  "media/final_ranking.png"):
        assert image in guide, image


def test_the_manual_sends_newcomers_to_the_guide_before_naming_policies():
    manual = read("docs/play/PLAYING_THE_GAME.md")
    first_id = min(manual.index(f"`{p}`") for p in pr.POLICY_IDS if f"`{p}`" in manual)
    assert manual.index("the guide") < first_id


def test_frozen_labels_and_decision_are_unchanged():
    for rel in ("README.md", "docs/research/RESULTS.md", "docs/research/STORY.md"):
        text = read(rel)
        assert pr.HYBRID_V1_STATUS in text and pr.HYBRID_V2_STATUS in text, rel
    assert "provider reduction is real but Hybrid-v2 is not safely promotable" in read("README.md")
    assert "not safely promotable under the frozen confirmatory rule" in read("docs/research/RESULTS.md")


def test_research_summary_and_personality_heterogeneity():
    readme = read("README.md")
    for needle in ("Local action fidelity is not closed-loop behavioural fidelity", "generalized better than pure B2",
                   "14.6%", "0.0182", "0.0168", "+0.0008"):
        assert needle in readme, needle
    for row in ("| cautious | 802 | 581 |", "| opportunist | 967 | 913 |", "| racer | 732 | 809 |",
                "| saboteur | 543 | 296 |"):
        assert row in readme, row
    assert "did not establish that imbalance" in readme and "causal mechanism" in readme


def test_the_reproducibility_boundary_is_explicit():
    for rel in ("README.md", "docs/research/RESULTS.md"):
        text = read(rel)
        assert "lightweight" in text and "distributed separately" in text, rel
    assert "not built or published yet" in read("docs/research/RESULTS.md")


def test_the_story_is_curated_by_idea():
    story = read("docs/research/STORY.md")
    for idea in ("Game foundations", "Partial information", "Memory", "The LLM controller", "Provider abstraction",
                 "Surrogate policies", "The closed-loop pathology", "Hybrid routing", "Environment shift",
                 "Hard-example distillation", "The playable runtime", "Gameplay polish"):
        assert re.search(rf"### \d+\. {re.escape(idea)}", story), idea
    assert "Milestone" not in story and "_SPEC" not in story


def test_the_claims_are_all_listed():
    results = read("docs/research/RESULTS.md")
    assert "33 claims: CONFIRMED 10" in results
    assert results.count("| `") >= 33


def test_assets_inventory_names_every_folder():
    inventory = read("assets/README.md")
    for folder in sorted(p.name for p in (ROOT / "assets").iterdir() if p.is_dir()):
        assert f"`{folder}/" in inventory, folder


def test_example_replays_are_documented():
    readme = read("examples/replays/README.md")
    bundles = sorted(p.name for p in (ROOT / "examples/replays").glob("*.json"))
    assert len(bundles) >= 6
    for name in bundles:
        assert f"`{name}`" in readme, name


def test_scientific_artifacts_are_byte_identical_to_the_frozen_hashes():
    arts = PROVENANCE["scientific_artifacts"]
    assert {a["id"] for a in arts} >= {"B1", "B2-v1", "B2-v2", "R2.1"}
    for a in arts:
        digest = "sha256:" + hashlib.sha256((ROOT / a["public_path"]).read_bytes()).hexdigest()
        assert digest == a["exported_sha256"] == a["source_sha256"], a["public_path"]
        if a["frozen_sha256"]:
            assert digest == a["frozen_sha256"], a["public_path"]


def test_no_private_archive_directory_and_no_archive_only_block():
    for rel in PROVENANCE["forbidden"]:
        if rel not in LOCAL_OUTPUTS:
            assert not (ROOT / rel).exists(), rel
    for page in ROOT.rglob("*.md"):
        assert "<!-- private:" not in page.read_text(encoding="utf-8"), page


def test_gitignore_keeps_user_traces_local_but_tracks_the_examples():
    lines = [ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    for anchored in ("/runs/", "/replays/", "/runtime_logs/"):
        assert anchored in lines, anchored
    assert "replays/" not in lines and "runs/" not in lines  # unanchored would also ignore examples/replays/
    for secret in (".env", ".env.*"):
        assert secret in lines


def test_no_dead_link_no_secret_no_local_path(validator):
    validator.check_links()
    secrets, local = validator.scan_tree()
    assert validator.results[-1]["status"] == "PASS", validator.results[-1]["detail"]
    assert secrets == [] and local == []
