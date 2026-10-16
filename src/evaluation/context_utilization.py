"""M5E.2 spec section 17: context/headroom metrics - a pure, read-only
analysis layer over telemetry M5E.1/M5E.1.1 already record (`input_tokens`
per `llm_decision`, `variant.num_ctx`). Never a new source of truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ContextUtilizationReport:
    configured_num_ctx: int | None
    max_input_tokens: int | None
    mean_input_tokens: float | None
    p95_input_tokens: float | None
    max_context_utilization_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured_num_ctx": self.configured_num_ctx,
            "max_input_tokens": self.max_input_tokens,
            "mean_input_tokens": self.mean_input_tokens,
            "p95_input_tokens": self.p95_input_tokens,
            "max_context_utilization_ratio": self.max_context_utilization_ratio,
        }


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile - no numpy dependency for one small stat."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def compute_context_utilization(input_tokens: list[int], configured_num_ctx: int | None) -> ContextUtilizationReport:
    """`input_tokens` is one variant/episode-group's real, Ollama-reported
    per-decision `input_tokens` values (never estimates - spec section 4's
    "never present an estimate as measured" carried over from M5E.1.1).
    `configured_num_ctx` is that variant's own `VariantSpec.num_ctx` (None
    means Ollama's own unconfigured default was in effect, in which case
    the utilization ratio can't be computed - there's nothing to divide
    by - and is honestly reported as None, not guessed)."""
    values = [t for t in input_tokens if t is not None]
    if not values:
        return ContextUtilizationReport(configured_num_ctx, None, None, None, None)
    max_tokens = max(values)
    mean_tokens = sum(values) / len(values)
    p95_tokens = _percentile([float(v) for v in values], 95)
    ratio = (max_tokens / configured_num_ctx) if configured_num_ctx else None
    return ContextUtilizationReport(
        configured_num_ctx=configured_num_ctx,
        max_input_tokens=max_tokens,
        mean_input_tokens=mean_tokens,
        p95_input_tokens=p95_tokens,
        max_context_utilization_ratio=ratio,
    )
