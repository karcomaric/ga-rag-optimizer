"""Shared evaluation and JSONL logging loop for the baseline optimizers.

Random search, grid search and the default-config baseline all reduce to
evaluating a sequence of candidates and logging one JSONL line per trial.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from src.schemas.fitness_result import FitnessFn
from src.schemas.optimizer_result import OptimizerResult

logger = logging.getLogger(__name__)


def run_trials(
    candidates: Iterable[dict[str, Any]],
    fitness_fn: FitnessFn,
    log_path: Path | None = None,
) -> OptimizerResult:
    """Evaluates a sequence of candidates in order and logs each trial.

    Opens and closes the log file, the evaluation itself runs in
    ``_evaluate_all``.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidates darf nicht leer sein.")

    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    start = time.perf_counter()
    try:
        best_index, best_fitness = _evaluate_all(candidates, fitness_fn, log_file)
    finally:
        if log_file is not None:
            log_file.close()
    wall = time.perf_counter() - start

    return OptimizerResult(
        best_individual=dict(candidates[best_index]),
        best_fitness=best_fitness,
        n_evaluations=len(candidates),
        runtime_s=wall,
    )


def _evaluate_all(
    candidates: list[dict[str, Any]],
    fitness_fn: FitnessFn,
    log_file: TextIO | None,
) -> tuple[int, float]:
    """Evaluates every candidate in order and returns index and value of the best.

    Writes the same JSONL fields as ``GeneticAlgorithm._evaluate`` except ``generation``. Ties in
    ``composite`` are broken in favor of the earliest evaluated candidate.
    """
    best_index = -1
    best_fitness = float("-inf")
    for eval_index, individual in enumerate(candidates):
        trial_start = time.perf_counter()
        metrics = fitness_fn(individual)
        trial_wall = time.perf_counter() - trial_start

        if log_file is not None:
            record: dict[str, Any] = {
                "eval_index": eval_index,
                "individual": dict(individual),
                "runtime_s": trial_wall,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            record.update(metrics)
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

        composite = float(metrics["composite"])
        if composite > best_fitness:
            best_fitness = composite
            best_index = eval_index
        logger.info(
            "Evaluation %d/%d: composite=%.4f, bestes bisher=%.4f",
            eval_index + 1,
            len(candidates),
            composite,
            best_fitness,
        )
    return best_index, best_fitness
