"""M5C/M5C.1: runtime registry/factory.

Deliberately tiny - one name-to-class mapping, no plugin framework (spec
section 8). Resolves `"native"` and (M5C.1) `"strands"`. An unknown runtime
name fails loudly with `ValueError`; it never silently substitutes another
runtime (spec section 7) - a caller who misspells or requests an
unimplemented runtime must find out immediately, not get a quietly
different experiment.

Importing this module never requires the `strands-agents` package: the
import below only reaches `StrandsAgentAdapter`'s class definition, and
that class only imports `strands` lazily inside `__init__` (see
strands_agent.py's `_import_strands`). Native must keep working when
Strands is unavailable (M5C.1 spec section 25); only actually selecting
`agent_runtime="strands"` without the SDK installed raises
`StrandsUnavailableError`.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentAdapter
from src.agents.native_agent import NativeAgentAdapter, NativeToolLikeAgentAdapter
from src.agents.strands_agent import StrandsAgentAdapter

DEFAULT_AGENT_RUNTIME = "native"

_RUNTIMES: dict[str, type[AgentAdapter]] = {
    "native": NativeAgentAdapter,
    "strands": StrandsAgentAdapter,
    # M5D.7/M5D.8: Native's tool-like representation (spec section 7 of
    # both) - identical NativeAgentAdapter/LLMController path under the
    # hood, only legal-action presentation differs. A real registry entry
    # (not a special-cased flag) so it flows through Game/EpisodeRunner/
    # VariantSpec exactly like the other two.
    "native_tool_like": NativeToolLikeAgentAdapter,
}


def create_agent_runtime(runtime_name: str | None, **kwargs: Any) -> AgentAdapter:
    name = (runtime_name or DEFAULT_AGENT_RUNTIME).lower()
    adapter_cls = _RUNTIMES.get(name)
    if adapter_cls is None:
        available = ", ".join(sorted(_RUNTIMES)) or "(none registered)"
        raise ValueError(f"Unknown agent_runtime {name!r}. Available runtimes: {available}.")
    return adapter_cls(**kwargs)
