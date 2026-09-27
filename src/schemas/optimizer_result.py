"""Result structure of a baseline optimizer run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptimizerResult:
    """Aggregated result of a completed baseline optimizer run.

    Structurally analogous to ``GAResult``, but without a logbook, since the
    baselines have no generational statistics.
    """

    best_individual: dict[str, Any]
    best_fitness: float
    n_evaluations: int
    runtime_s: float
