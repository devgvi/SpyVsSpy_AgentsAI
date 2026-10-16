"""Turns input into GameAction objects.

HumanController is one implementation of this interface. A future
AIController will implement `get_action(observation) -> GameAction` and can
be swapped in without the game loop or game rules changing at all.
"""

import math

import pygame

from src import constants
from src.actions import ActionType, GameAction

TRAP_SELECT_KEYS = {
    pygame.K_1: "bomb",
    pygame.K_2: "electric",
    pygame.K_3: "snare",
    pygame.K_4: "alarm",
}

MOVE_KEYS = {
    pygame.K_w: (0, -1),
    pygame.K_UP: (0, -1),
    pygame.K_s: (0, 1),
    pygame.K_DOWN: (0, 1),
    pygame.K_a: (-1, 0),
    pygame.K_LEFT: (-1, 0),
    pygame.K_d: (1, 0),
    pygame.K_RIGHT: (1, 0),
}


class HumanController:
    """Reads keyboard state and produces exactly one GameAction per frame.

    `selected_trap_type` is UI/input state, not gameplay state - it only
    decides which trap_type rides along on a PLACE_TRAP GameAction, the
    same way this class already decides which direction rides along on a
    MOVE action. GameState never reads it directly.
    """

    def __init__(self) -> None:
        self.selected_trap_type = constants.DEFAULT_TRAP_TYPE

    def get_action(
        self,
        events: list[pygame.event.Event],
        keys: pygame.key.ScancodeWrapper,
    ) -> GameAction:
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue

            # Edge-triggered: a single KEYDOWN event, not a held key, so
            # holding a key does not repeat the action every frame.
            if event.key == pygame.K_e:
                return GameAction(ActionType.INTERACT)
            if event.key == pygame.K_i:
                return GameAction(ActionType.INSPECT)
            if event.key == pygame.K_t:
                return GameAction(ActionType.PLACE_TRAP, trap_type=self.selected_trap_type)
            if event.key == pygame.K_d:
                return GameAction(ActionType.DISARM)
            if event.key in TRAP_SELECT_KEYS:
                self.selected_trap_type = TRAP_SELECT_KEYS[event.key]
                # Selecting a trap type is not itself a game action - keep
                # scanning this frame's events for one.

        direction = self._movement_direction(keys)
        if direction == (0.0, 0.0):
            return GameAction(ActionType.WAIT)
        return GameAction(ActionType.MOVE, direction=direction)

    @staticmethod
    def _movement_direction(keys: pygame.key.ScancodeWrapper) -> tuple[float, float]:
        dx = 0.0
        dy = 0.0
        for key, (kx, ky) in MOVE_KEYS.items():
            if keys[key]:
                dx += kx
                dy += ky

        magnitude = math.hypot(dx, dy)
        if magnitude == 0:
            return (0.0, 0.0)
        return (dx / magnitude, dy / magnitude)
