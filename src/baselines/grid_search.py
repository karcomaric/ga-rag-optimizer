"""Grid search baseline over the full hyperparameter search space."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.baselines.trial_eval import run_trials
from src.ga_engine.search_space import SearchSpace
from src.schemas.fitness_result import FitnessFn
from src.schemas.optimizer_result import OptimizerResult


class GridSearchOptimizer:
    """Enumerates the full Cartesian product of the search space.

    The order is deterministic and needs no seed, because ``SearchSpace``
    sorts int-valued genes and keeps the YAML order for str-valued ones.
    Candidates violating ``chunk_overlap < chunk_size`` are dropped instead of
    repaired, so no combination appears twice.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        fitness_fn: FitnessFn,
        log_path: Path | None = None,
        max_trials: int | None = None,
    ) -> None:
        """Initializes the optimizer.

        ``max_trials`` caps the number of valid candidates and exists for smoke
        runs, ``None`` enumerates the full valid grid.
        """
        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.log_path = log_path
        self.max_trials = max_trials

    def run(self) -> OptimizerResult:
        """Enumerates and evaluates the (possibly capped) valid grid."""
        candidates: list[dict[str, Any]] = []
        for combination in self.search_space.full_grid():
            if self.max_trials is not None and len(candidates) >= self.max_trials:
                break
            if self.search_space.is_valid(combination):
                candidates.append(combination)

        return run_trials(candidates, self.fitness_fn, self.log_path)
