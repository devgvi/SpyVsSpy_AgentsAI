"""Loads mission item definitions from data/items.json.

Which furniture contains which item lives in the level data
(GameObject.contains), not here - this module only defines what an item
*is* (its display name and icon).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Item:
    id: str
    name: str
    sprite: str | None


@dataclass
class ItemCatalog:
    items: dict[str, Item]
    required: list[str]
    # M4B: display names for knowledge ids (e.g. "safe_code" -> "Safe
    # Code") - never a sprite, since knowledge is never rendered as an
    # inventory icon (Milestone_4.md: "codes are knowledge, NEVER
    # inventory").
    knowledge: dict[str, str] = field(default_factory=dict)

    def knowledge_name(self, knowledge_id: str) -> str:
        return self.knowledge.get(knowledge_id, knowledge_id)


def load_items(path: str) -> ItemCatalog:
    data = json.loads(Path(path).read_text())

    items = {
        entry["id"]: Item(id=entry["id"], name=entry["name"], sprite=entry.get("sprite"))
        for entry in data["items"]
    }

    knowledge = {entry["id"]: entry["name"] for entry in data.get("knowledge", [])}

    return ItemCatalog(items=items, required=list(data["required"]), knowledge=knowledge)
