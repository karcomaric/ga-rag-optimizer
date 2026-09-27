"""Result structure of a GA run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GAResult:
    """Aggregated result of a complete GA run.

    ``logbook`` holds one dict per generation with the generation number, the
    number of evaluations and the registered statistics, ``n_evaluations``
    counts the fitness calls actually made.
    """

    best_individual: dict[str, Any]
    best_fitness: float
    logbook: list[dict[str, Any]]
    n_evaluations: int
    runtime_s: float
