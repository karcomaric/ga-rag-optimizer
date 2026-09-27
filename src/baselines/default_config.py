"""Default-configuration baseline, a single fixed-hyperparameter evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.baselines.trial_eval import run_trials
from src.schemas.fitness_result import FitnessFn
from src.schemas.optimizer_result import OptimizerResult


class DefaultConfigBaseline:
    """Evaluates exactly one fixed individual, the LangChain default configuration.

    The class does not know about the YAML config, it receives the
    already-built individual dict from the caller.
    """

    def __init__(
        self,
        default_individual: dict[str, Any],
        fitness_fn: FitnessFn,
        log_path: Path | None = None,
    ) -> None:
        """Initializes the baseline with a fixed individual, ``log_path`` may be None."""
        self.default_individual = default_individual
        self.fitness_fn = fitness_fn
        self.log_path = log_path

    def run(self) -> OptimizerResult:
        """Evaluates the default individual once."""
        return run_trials([self.default_individual], self.fitness_fn, self.log_path)
