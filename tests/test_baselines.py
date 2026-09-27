"""Tests for `src.baselines`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.baselines import (
    DefaultConfigBaseline,
    GridSearchOptimizer,
    RandomSearchOptimizer,
)
from src.baselines.trial_eval import run_trials
from src.ga_engine import FitnessResult, SearchSpace

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_config.yaml"

REDUCED_SPACE = {
    "chunk_size": [256, 512, 1024],
    "chunk_overlap": [0, 64],
    "top_k": [3, 5, 10],
    "embedding_model": ["all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"],
    "retrieval_strategy": ["dense", "bm25", "hybrid"],
}


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        {
            "chunk_size": [128, 512, 1024],
            "chunk_overlap": [0, 32, 64, 128],
            "top_k": [2, 3, 5, 10],
            "embedding_model": ["m1", "m2", "m3"],
            "retrieval_strategy": ["dense", "bm25", "hybrid"],
        }
    )


def _deterministic_fitness(ind: dict[str, Any]) -> FitnessResult:
    """Deterministic fitness, prefers large chunk_size and top_k."""
    score = (ind["chunk_size"] / 1024.0) * 0.5 + (ind["top_k"] / 10.0) * 0.5
    return FitnessResult(
        composite=score,
        context_recall=score,
        context_precision=score,
        answer_f1=score,
        exact_match=score,
        per_question=[],
    )


def test_random_search_runs_exact_n_trials(space: SearchSpace, tmp_path: Path) -> None:
    log_path = tmp_path / "random.jsonl"
    optimizer = RandomSearchOptimizer(
        search_space=space,
        fitness_fn=_deterministic_fitness,
        n_trials=10,
        seed=42,
        log_path=log_path,
    )
    result = optimizer.run()

    assert result.n_evaluations == 10
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10


def test_random_search_is_reproducible_with_same_seed(space: SearchSpace, tmp_path: Path) -> None:
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    result_a = RandomSearchOptimizer(
        space, _deterministic_fitness, 10, seed=42, log_path=log_a
    ).run()
    result_b = RandomSearchOptimizer(
        space, _deterministic_fitness, 10, seed=42, log_path=log_b
    ).run()

    individuals_a = [json.loads(line)["individual"] for line in log_a.read_text().splitlines()]
    individuals_b = [json.loads(line)["individual"] for line in log_b.read_text().splitlines()]
    assert individuals_a == individuals_b
    assert result_a.best_individual == result_b.best_individual


def test_random_search_differs_with_different_seed(space: SearchSpace, tmp_path: Path) -> None:
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    RandomSearchOptimizer(space, _deterministic_fitness, 10, seed=42, log_path=log_a).run()
    RandomSearchOptimizer(space, _deterministic_fitness, 10, seed=4711, log_path=log_b).run()

    individuals_a = [json.loads(line)["individual"] for line in log_a.read_text().splitlines()]
    individuals_b = [json.loads(line)["individual"] for line in log_b.read_text().splitlines()]
    assert individuals_a != individuals_b


def test_grid_search_enumerates_full_valid_space(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    full_space = SearchSpace(config["search_space"])
    log_path = tmp_path / "grid.jsonl"
    optimizer = GridSearchOptimizer(full_space, _deterministic_fitness, log_path=log_path)
    result = optimizer.run()

    assert result.n_evaluations == 540
    individuals = [json.loads(line)["individual"] for line in log_path.read_text().splitlines()]
    unique_individuals = {tuple(sorted(ind.items())) for ind in individuals}
    assert len(unique_individuals) == 540
    assert all(ind["chunk_overlap"] < ind["chunk_size"] for ind in individuals)


def test_grid_search_order_is_deterministic(tmp_path: Path) -> None:
    reduced_space = SearchSpace(REDUCED_SPACE)
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    GridSearchOptimizer(reduced_space, _deterministic_fitness, log_path=log_a).run()
    GridSearchOptimizer(reduced_space, _deterministic_fitness, log_path=log_b).run()

    individuals_a = [json.loads(line)["individual"] for line in log_a.read_text().splitlines()]
    individuals_b = [json.loads(line)["individual"] for line in log_b.read_text().splitlines()]
    assert individuals_a == individuals_b


def test_grid_search_max_trials_caps_candidates(tmp_path: Path) -> None:
    reduced_space = SearchSpace(REDUCED_SPACE)
    log_path = tmp_path / "grid.jsonl"
    optimizer = GridSearchOptimizer(
        reduced_space, _deterministic_fitness, log_path=log_path, max_trials=5
    )
    result = optimizer.run()

    assert result.n_evaluations == 5


def test_default_config_evaluates_exactly_once(tmp_path: Path) -> None:
    default_individual = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4,
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_strategy": "dense",
    }
    log_path = tmp_path / "default.jsonl"
    baseline = DefaultConfigBaseline(default_individual, _deterministic_fitness, log_path=log_path)
    result = baseline.run()

    assert result.n_evaluations == 1
    assert result.best_individual == default_individual


def test_jsonl_schema_matches_ga_history(space: SearchSpace, tmp_path: Path) -> None:
    log_path = tmp_path / "random.jsonl"
    RandomSearchOptimizer(space, _deterministic_fitness, 5, seed=42, log_path=log_path).run()

    first = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    for key in (
        "eval_index",
        "individual",
        "composite",
        "context_recall",
        "context_precision",
        "answer_f1",
        "exact_match",
        "per_question",
        "runtime_s",
        "timestamp",
    ):
        assert key in first, f"Pflichtfeld {key!r} fehlt im JSONL-Record."


def test_run_trials_empty_candidates_raises() -> None:
    with pytest.raises(ValueError, match="candidates"):
        run_trials([], _deterministic_fitness, log_path=None)
