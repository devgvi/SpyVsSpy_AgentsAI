"""The action interface shared by every controller (human or AI).

Nothing in the game simulation may bypass a GameAction. A controller's only
job is to look at input (keyboard, or later, an AI observation) and produce
one of these.
"""

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    MOVE = "move"
    INTERACT = "interact"  # search a searchable object; open a door; attempt the exit
    INSPECT = "inspect"
    PLACE_TRAP = "place_trap"
    DISARM = "disarm"
    WAIT = "wait"
    SEND_MESSAGE = "send_message"  # M5F.2: target_id is the recipient actor id


@dataclass
class GameAction:
    type: ActionType
    direction: tuple[float, float] | None = None
    target_id: str | None = None
    trap_type: str | None = None  # which trap to place; only used by PLACE_TRAP
    message: str | None = None  # the text to send; only used by SEND_MESSAGE
    # M5F.2.2: an optional message that rides along with ANY action type
    # (the "zero decision cost" envelope - spec section 19's "decision
    # envelope: optional communication event + ordinary physical
    # GameAction", never a new ActionType). Shape: {"recipient_id": str,
    # "text": str}. Mutually distinct from `message` above - that field is
    # only ever used by the standalone SEND_MESSAGE action itself.
    accompanying_message: dict | None = None
    # live riddle puzzle (src/puzzles.py): the code entered at the safe -
    # INTERACT on the safe with a code is "enter_safe_code"
    code: str | None = None


@dataclass
class InteractionResult:
    """What happened when an INTERACT/INSPECT/PLACE_TRAP/DISARM action was applied.

    Game rules return this instead of touching the renderer directly, so
    the same structured record can later be handed to an AI agent as
    feedback on the action it just took.
    """

    success: bool
    message: str
    item_found: str | None = None
    trap_detected: str | None = None
    trap_triggered: str | None = None
