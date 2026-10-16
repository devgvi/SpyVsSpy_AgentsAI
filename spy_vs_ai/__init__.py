"""Spy vs AI - the playable multi-policy / multi-provider runtime (FINAL-2).

A user-facing facade over the existing game and the frozen research policies:

    python -m spy_vs_ai.play      # a single visual (or --headless) match, each spy configured independently
    python -m spy_vs_ai.replay    # an offline replay of a saved match (0 provider calls)

Run both from the project root (the directory that holds src/, workshop/ and this package).

Nothing in this package changes the science: the engine (src/), the live game (workshop/live/) and every
frozen research artifact (workshop/policies/, workshop/policy_data/, workshop/policy_reports/) are used read-only.
Frozen policies are wrapped, never edited; runtime outputs go to runs/ (never into the research archive).
"""
from __future__ import annotations

__version__ = "final2-runtime-v1"
RUNTIME_SCHEMA = "spy-vs-ai-runtime-v1"
