"""Minimal mission-item inventory. No duplicates."""

from dataclasses import dataclass, field


@dataclass
class Inventory:
    items: set[str] = field(default_factory=set)

    def add(self, item_id: str) -> bool:
        """Add an item. Returns True if it was newly added, False if already held."""
        if item_id in self.items:
            return False
        self.items.add(item_id)
        return True

    def has(self, item_id: str) -> bool:
        return item_id in self.items

    def missing(self, required: list[str]) -> list[str]:
        return [item_id for item_id in required if item_id not in self.items]

    def has_all(self, required: list[str]) -> bool:
        return all(item_id in self.items for item_id in required)
