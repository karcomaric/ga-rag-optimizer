"""Tests for the genetic operators of ``src.ga_engine.ga_loop``."""

from __future__ import annotations

import random

import pytest

from src.ga_engine.ga_loop import GeneticAlgorithm
from src.ga_engine.search_space import SearchSpace


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


def test_crossover_preserves_all_gene_keys(space: SearchSpace) -> None:
    random.seed(1)
    parent_a = space.sample()
    parent_b = space.sample()

    child_a, child_b = GeneticAlgorithm.uniform_crossover(parent_a, parent_b, space, indpb=0.5)

    assert child_a.keys() == parent_a.keys()
    assert child_b.keys() == parent_b.keys()


def test_crossover_does_not_mutate_parents(space: SearchSpace) -> None:
    random.seed(2)
    parent_a = space.sample()
    parent_b = space.sample()
    snapshot_a = dict(parent_a)
    snapshot_b = dict(parent_b)

    GeneticAlgorithm.uniform_crossover(parent_a, parent_b, space, indpb=0.5)

    assert parent_a == snapshot_a
    assert parent_b == snapshot_b


def test_crossover_is_reproducible_with_seed(space: SearchSpace) -> None:
    random.seed(7)
    parent_a = space.sample()
    parent_b = space.sample()

    random.seed(99)
    a1, b1 = GeneticAlgorithm.uniform_crossover(parent_a, parent_b, space, indpb=0.5)
    random.seed(99)
    a2, b2 = GeneticAlgorithm.uniform_crossover(parent_a, parent_b, space, indpb=0.5)

    assert a1 == a2
    assert b1 == b2


def test_crossover_never_produces_invalid_overlap(space: SearchSpace) -> None:
    random.seed(11)
    for _ in range(200):
        parent_a = space.sample()
        parent_b = space.sample()
        child_a, child_b = GeneticAlgorithm.uniform_crossover(parent_a, parent_b, space, indpb=0.5)
        assert child_a["chunk_overlap"] < child_a["chunk_size"]
        assert child_b["chunk_overlap"] < child_b["chunk_size"]


def test_mutate_with_indpb_one_changes_all_genes(space: SearchSpace) -> None:
    random.seed(3)
    individual = space.sample()

    mutant = GeneticAlgorithm.mutate_individual(individual, space, indpb=1.0)

    for gene in individual:
        assert mutant[gene] != individual[gene] or gene == "chunk_overlap"


def test_mutate_with_indpb_zero_changes_nothing(space: SearchSpace) -> None:
    random.seed(4)
    individual = space.sample()

    mutant = GeneticAlgorithm.mutate_individual(individual, space, indpb=0.0)

    assert mutant == individual
    assert mutant is not individual


def test_mutate_is_reproducible_with_seed(space: SearchSpace) -> None:
    random.seed(5)
    individual = space.sample()

    random.seed(123)
    m1 = GeneticAlgorithm.mutate_individual(individual, space, indpb=0.5)
    random.seed(123)
    m2 = GeneticAlgorithm.mutate_individual(individual, space, indpb=0.5)

    assert m1 == m2


def test_mutate_never_produces_invalid_overlap(space: SearchSpace) -> None:
    random.seed(6)
    for _ in range(200):
        individual = space.sample()
        mutant = GeneticAlgorithm.mutate_individual(individual, space, indpb=1.0)
        assert mutant["chunk_overlap"] < mutant["chunk_size"]
