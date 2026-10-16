"""Spy vs AI agent-engineering workshop (M7).

Nine zero-model-call labs built on stored M5/M6 evidence. Run them from
the project root (the directory that contains `src/` and `workshop/`):

    python -m workshop.scripts.preflight
    python -m workshop.labs.lab01 --session <your-name>

The live visual game (real LLM-controlled spies, provider access needed):

    python -m workshop.play --provider bedrock --model <model-id>

Importing this package first keeps the labs quiet and side-effect free:
no bytecode caches are written next to the source tree, pygame gets the
headless SDL drivers, and its support banner is suppressed.
`workshop.play` opens a real window, so it undoes the headless video
default when (and only when) this package was the one that set it.
"""

import os
import sys

# Which SDL defaults this import applied itself (True) vs found already set
# by the user (False) - read by workshop.play before it opens a window.
SDL_DEFAULTS_APPLIED = {name: name not in os.environ for name in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER")}

sys.dont_write_bytecode = True
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
