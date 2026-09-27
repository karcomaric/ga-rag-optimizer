"""Random search baseline over the full hyperparameter search space."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.trial_eval import run_trials
from src.ga_engine.search_space import SearchSpace
from src.schemas.fitness_result import FitnessFn
from src.schemas.optimizer_result import OptimizerResult


class RandomSearchOptimizer:
    """Draws `n_trials` random candidates from the search space and evaluates them.

    Sampling is with replacement, which is standard for random search.
    Repeated configurations are evaluated again in full, there is no fitness
    cache.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        fitness_fn: FitnessFn,
        n_trials: int,
        seed: int,
        log_path: Path | None = None,
    ) -> None:
        """Initializes the optimizer and seeds ``random`` and ``numpy``.

        ``log_path`` of ``None`` disables the JSONL log.
        """
        random.seed(seed)
        np.random.seed(seed)

        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.n_trials = n_trials
        self.log_path = log_path

    def run(self) -> OptimizerResult:
        """Draws and evaluates ``n_trials`` random candidates."""
        candidates: list[dict[str, Any]] = []
        for _ in range(self.n_trials):
            candidates.append(self.search_space.sample())
        return run_trials(candidates, self.fitness_fn, self.log_path)
