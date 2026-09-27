"""Tests for ``src.ga_engine.ga_loop``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.ga_engine import FitnessResult, GeneticAlgorithm, SearchSpace


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


def _build_ga(space: SearchSpace, seed: int, log_path: Path | None = None) -> GeneticAlgorithm:
    return GeneticAlgorithm(
        search_space=space,
        fitness_fn=_deterministic_fitness,
        population_size=4,
        n_generations=3,
        crossover_prob=0.7,
        mutation_prob=0.3,
        tournament_size=2,
        mutation_indpb=0.2,
        crossover_indpb=0.5,
        seed=seed,
        log_path=log_path,
    )


def test_ga_run_returns_result(space: SearchSpace) -> None:
    ga = _build_ga(space, seed=42)
    result = ga.run()

    assert isinstance(result.best_individual, dict)
    assert set(result.best_individual.keys()) == set(space.gene_names)
    assert 0.0 <= result.best_fitness <= 1.0
    assert result.n_evaluations >= 4
    assert result.n_evaluations <= 4 * (1 + 3)
    assert result.runtime_s > 0.0
    assert len(result.logbook) == 4


def test_logbook_records_genotypic_diversity_per_generation(space: SearchSpace) -> None:
    result = _build_ga(space, seed=42).run()

    for record in result.logbook:
        assert 1 <= record["unique_genomes"] <= 4
        assert 1.0 <= record["gene_diversity"] <= 4.0


def test_ga_is_reproducible_with_same_seed(space: SearchSpace) -> None:
    result_a = _build_ga(space, seed=42).run()
    result_b = _build_ga(space, seed=42).run()

    assert result_a.best_individual == result_b.best_individual
    assert result_a.best_fitness == result_b.best_fitness


def test_ga_is_different_with_different_seed(space: SearchSpace) -> None:
    result_a = _build_ga(space, seed=42).run()
    result_b = _build_ga(space, seed=4711).run()

    assert result_a.logbook != result_b.logbook


def test_jsonl_log_contains_one_line_per_evaluation(space: SearchSpace, tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    ga = _build_ga(space, seed=42, log_path=log_path)
    result = ga.run()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == result.n_evaluations

    first = json.loads(lines[0])
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
        "generation",
    ):
        assert key in first, f"Pflichtfeld {key!r} fehlt im JSONL-Record."


def test_jsonl_log_generation_matches_logbook(space: SearchSpace, tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    ga = _build_ga(space, seed=42, log_path=log_path)
    result = ga.run()

    generations = [json.loads(line)["generation"] for line in log_path.read_text().splitlines()]

    assert generations[:4] == [0, 0, 0, 0]
    assert generations == sorted(generations)
    expected_counts = {record["gen"]: record["nevals"] for record in result.logbook}
    actual_counts = {gen: generations.count(gen) for gen in set(generations)}
    assert actual_counts == {gen: count for gen, count in expected_counts.items() if count > 0}


def test_all_evaluated_individuals_respect_overlap_constraint(
    space: SearchSpace, tmp_path: Path
) -> None:
    log_path = tmp_path / "log.jsonl"
    ga = _build_ga(space, seed=42, log_path=log_path)
    ga.run()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        individual = json.loads(line)["individual"]
        assert individual["chunk_overlap"] < individual["chunk_size"]
